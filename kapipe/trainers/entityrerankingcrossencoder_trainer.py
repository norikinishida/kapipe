import copy
import logging
import os

import numpy as np
import torch
# from apex import amp
import jsonlines
from tqdm import tqdm

from . import shared_functions
from .. import evaluation
from .. import utils
from ..utils import BestScoreHolder


logger = logging.getLogger(__name__)


class EntityRerankingCrossEncoderTrainer:

    def __init__(self, base_output_path):
        """
        Parameters
        ----------
        base_output_path : str
        """
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self):
        """
        Returns
        -------
        dict[str, str]
        """
        paths = {}

        # Path to config file
        paths["path_config"] = self.base_output_path + "/config"
        # Path to model snapshot
        paths["path_snapshot"] = self.base_output_path + "/model"

        # Paths to training-losses and validation-scores files
        paths["path_train_losses"] = self.base_output_path + "/train.losses.jsonl"
        paths["path_dev_evals"] = self.base_output_path + "/dev.eval.jsonl"

        # Paths to validation outputs and scores
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"

        # Paths to evaluation outputs and scores
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        return paths

    def setup_dataset(self, system, documents, candidate_entities, split):
        """
        Parameters
        ----------
        system : EntityRerankingCrossEncoderSystem
        documents : list[Document]
        candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        split : str
        """
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(system.entity_dict.keys()))
            gold_documents = []
            for document, candidate_entities_for_doc in tqdm(
                zip(documents, candidate_entities),
                desc="dataset setup"
            ):
                gold_doc = copy.deepcopy(document)

                cands_for_mentions \
                    = candidate_entities_for_doc["candidate_entities"]
                mentions = document["mentions"]
                assert len(mentions) == len(cands_for_mentions)

                for m_i, (mention, cands_for_mention) in enumerate(zip(
                    mentions,
                    cands_for_mentions
                )):
                    cand_entity_ids = [
                        c["entity_id"] for c in cands_for_mention
                    ]
                    entity_id = mention["entity_id"]
                    in_kb = entity_id in kb_entity_ids
                    in_cand = entity_id in cand_entity_ids
                    gold_doc["mentions"][m_i]["in_kb"] = in_kb
                    gold_doc["mentions"][m_i]["in_cand"] = in_cand
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def train(
        self,
        system,
        train_documents,
        train_candidate_entities,
        dev_documents,
        dev_candidate_entities
    ):
        """
        Parameters
        ----------
        system : EntityRerankingCrossEncoderSystem
        train_documents : list[Document]
        train_candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        dev_documents : list[Document]
        dev_candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        """
        train_doc_indices = np.arange(len(train_documents))
        train_doc_index_and_mention_index_tuples = []
        for doc_i in train_doc_indices:
            for m_i in range(len(train_documents[doc_i]["mentions"])):
                train_doc_index_and_mention_index_tuples.append((doc_i, m_i))
        train_doc_index_and_mention_index_tuples \
            = np.asarray(train_doc_index_and_mention_index_tuples)

        ##################
        # Get optimizer and scheduler
        ##################

        n_train = len(train_doc_index_and_mention_index_tuples)
        max_epoch = system.config["max_epoch"]
        batch_size = system.config["batch_size"]
        gradient_accumulation_steps \
            = system.config["gradient_accumulation_steps"]
        total_update_steps \
            = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * system.config["warmup_ratio"])

        logger.info("Number of training mentions: %d" % n_train)
        logger.info("Number of epochs: %d" % max_epoch)
        logger.info("Batch size: %d" % batch_size)
        logger.info("Gradient accumulation steps: %d" % gradient_accumulation_steps)
        logger.info("Total update steps: %d" % total_update_steps)
        logger.info("Warmup steps: %d" % warmup_steps)

        optimizer = shared_functions.get_optimizer2(
            model=system.model,
            config=system.config
        )
        # system.model, optimizer = amp.initialize(
        #     system.model,
        #     optimizer,
        #     opt_level="O1",
        #     verbosity=0
        # )
        scheduler = shared_functions.get_scheduler2(
            optimizer=optimizer,
            total_update_steps=total_update_steps,
            warmup_steps=warmup_steps
        )

        ##################
        # Get reporter and best score holder
        ##################

        writer_train = jsonlines.Writer(
            open(self.paths["path_train_losses"], "w"),
            flush=True
        )
        writer_dev = jsonlines.Writer(
            open(self.paths["path_dev_evals"], "w"),
            flush=True
        )
        bestscore_holder = BestScoreHolder(scale=1.0)
        bestscore_holder.init()

        ##################
        # Evaluate
        ##################

        scores = self.evaluate(
            system=system,
            documents=dev_documents,
            candidate_entities=dev_candidate_entities,
            split="dev",
            #
            get_scores_only=True
        )
        scores["epoch"] = 0
        scores["step"] = 0
        writer_dev.write(scores)
        logger.info(utils.pretty_format_dict(scores))

        bestscore_holder.compare_scores(
            scores["inkb_normalized_accuracy"]["accuracy"],
            0
        )

        ##################
        # Save
        ##################

        # Save the model
        system.save_model(path=self.paths["path_snapshot"])
        logger.info("Saved model to %s" % self.paths["path_snapshot"])

        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], system.config)
        utils.write_json(self.paths["path_config"], system.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

        ##################
        # Training-and-validation loops
        ##################

        bert_param, task_param = system.model.get_params()
        system.model.zero_grad()
        step = 0
        batch_i = 0

        # Variables for reporting
        loss_accum = 0.0
        acc_accum = 0.0
        accum_count = 0

        progress_bar = tqdm(total=total_update_steps, desc="training steps")
        for epoch in range(1, max_epoch + 1):

            perm = np.random.permutation(n_train)

            for instance_i in range(0, n_train, batch_size):

                ##################
                # Forward
                ##################

                batch_i += 1

                # Initialize loss
                batch_loss = 0.0
                batch_acc = 0.0
                actual_batchsize = 0

                for (doc_i, mention_index) in (
                    train_doc_index_and_mention_index_tuples[
                        perm[instance_i : instance_i + batch_size]
                    ]
                ):
                    doc_i = int(doc_i)
                    mention_index = int(mention_index)

                    # Forward and compute loss
                    one_loss, one_acc = system.compute_loss(
                        document=train_documents[doc_i],
                        candidate_entities_for_doc=\
                            train_candidate_entities[doc_i],
                        mention_index=mention_index
                    )

                    # Accumulate the loss
                    batch_loss = batch_loss + one_loss
                    batch_acc = batch_acc + one_acc
                    actual_batchsize += 1

                # Average the loss
                actual_batchsize = float(actual_batchsize)
                batch_loss = batch_loss / actual_batchsize # loss per mention
                batch_acc = batch_acc / actual_batchsize # accuracy per mention

                ##################
                # Backward
                ##################

                batch_loss = batch_loss / gradient_accumulation_steps
                batch_loss.backward()
                # with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()

                # Accumulate for reporting
                loss_accum += float(batch_loss.cpu())
                acc_accum += batch_acc
                accum_count += 1

                if batch_i % gradient_accumulation_steps == 0:

                    ##################
                    # Update
                    ##################

                    if system.config["max_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            bert_param,
                            system.config["max_grad_norm"]
                        )
                        torch.nn.utils.clip_grad_norm_(
                            task_param,
                            system.config["max_grad_norm"]
                        )
                        # torch.nn.utils.clip_grad_norm_(
                        #     amp.master_params(optimizer),
                        #     system.config["max_grad_norm"]
                        # )
                    optimizer.step()
                    scheduler.step()

                    system.model.zero_grad()
                    step += 1
                    progress_bar.update()
                    progress_bar.refresh()

                if (
                    (instance_i + batch_size >= n_train)
                    or
                    (
                        (batch_i % gradient_accumulation_steps == 0)
                        and
                        (step % system.config["n_steps_for_monitoring"] == 0)
                    )
                ):

                    ##################
                    # Report
                    ##################

                    out = {
                        "step": step,
                        "epoch": epoch,
                        "step_progress": "%d/%d" % (step, total_update_steps),
                        "step_progress(ratio)": \
                            float(step) / total_update_steps * 100.0,
                        "one_epoch_progress": \
                            "%d/%d" % (instance_i + actual_batchsize, n_train),
                        "one_epoch_progress(ratio)": (
                            float(instance_i + actual_batchsize)
                            / n_train
                            * 100.0
                        ),
                        "loss": loss_accum / accum_count,
                        "accuracy": 100.0 * acc_accum / accum_count,
                        "max_valid_inkb_acc": \
                            bestscore_holder.best_score,
                        "patience": bestscore_holder.patience
                    }
                    writer_train.write(out)
                    logger.info(utils.pretty_format_dict(out))
                    loss_accum = 0.0
                    acc_accum = 0.0
                    accum_count = 0

                if (
                    (instance_i + batch_size >= n_train)
                    or
                    (
                        (batch_i % gradient_accumulation_steps == 0)
                        and
                        (system.config["n_steps_for_validation"] > 0)
                        and
                        (step % system.config["n_steps_for_validation"] == 0)
                    )
                ):

                    ##################
                    # Evaluate
                    ##################

                    scores = self.evaluate(
                        system=system,
                        documents=dev_documents,
                        candidate_entities=dev_candidate_entities,
                        split="dev",
                        #
                        get_scores_only=True
                    )
                    scores["epoch"] = epoch
                    scores["step"] = step
                    writer_dev.write(scores)
                    logger.info(utils.pretty_format_dict(scores))

                    did_update = bestscore_holder.compare_scores(
                        scores["inkb_accuracy"]["accuracy"],
                        epoch
                    )
                    logger.info("[Step %d] Max validation InKB normalized accuracy: %f" % (step, bestscore_holder.best_score))

                    ##################
                    # Save
                    ##################

                    if did_update:
                        system.save_model(path=self.paths["path_snapshot"])
                        logger.info("Saved model to %s" % self.paths["path_snapshot"])

                    if (
                        bestscore_holder.patience
                        >= system.config["max_patience"]
                    ):
                        writer_train.close()
                        writer_dev.close()
                        progress_bar.close()
                        return

        writer_train.close()
        writer_dev.close()
        progress_bar.close()

    def evaluate(
        self,
        system,
        documents,
        candidate_entities,
        split,
        #
        get_scores_only=False,
    ):
        """
        Parameters
        ----------
        system : EntityRerankingCrossEncoderSystem
        documents : list[Document]
        candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        split : str
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # (documents, candidate_entities) -> path_pred
        result_documents = system.batch_extract(
            documents=documents,
            candidate_entities=candidate_entities
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        # (path_pred, path_gold) -> scores
        scores = evaluation.ed.accuracy(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        )
        scores.update(evaluation.ed.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        ))
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

