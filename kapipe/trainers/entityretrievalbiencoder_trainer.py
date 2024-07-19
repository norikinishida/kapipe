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


class EntityRetrievalBiEncoderTrainer:

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
        paths["path_dev_pred_retrieval"] = self.base_output_path + "/dev.pred_candidate_entities.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"

        # Paths to evaluation outputs and scores
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_pred_retrieval"] = self.base_output_path + "/test.pred_candidate_entities.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        # For the reranking-model training in the later stage,
        #   we need to annotate candidate entities also for the training set
        paths["path_train_pred"] = self.base_output_path + "/train.pred.json"
        paths["path_train_pred_retrieval"] = self.base_output_path + "/train.pred_candidate_entities.json"

        return paths

    def setup_dataset(self, system, documents, split):
        """
        Parameters
        ----------
        system : EntityRetrievalBiEncoderSystem
        documents : list[Document]
        split : str
        """
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(system.entity_dict.keys()))
            gold_documents = []
            for document in tqdm(documents, desc="dataset setup"):
                gold_doc = copy.deepcopy(document)
                for m_i, mention in enumerate(document["mentions"]):
                    in_kb = mention["entity_id"] in kb_entity_ids
                    gold_doc["mentions"][m_i]["in_kb"] = in_kb
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def train(
        self,
        system,
        train_documents,
        dev_documents,
    ):
        """
        Parameters
        ----------
        system : EntityRetrievalBiEncoderSystem
        train_documents : list[Document]
        dev_documents : list[Document]
        """
        train_doc_indices = np.arange(len(train_documents))

        ##################
        # Get optimizer and scheduler
        ##################

        n_train = len(train_doc_indices)
        max_epoch = system.config["max_epoch"]
        batch_size = system.config["batch_size"]
        gradient_accumulation_steps \
            = system.config["gradient_accumulation_steps"]
        total_update_steps \
            = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * system.config["warmup_ratio"])

        logger.info("Number of training documents: %d" % n_train)
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

        system.make_index()

        scores = self.evaluate(
            system=system,
            documents=dev_documents,
            split="dev",
            #
            get_scores_only=True
        )
        scores["epoch"] = 0
        scores["step"] = 0
        writer_dev.write(scores)
        logger.info(utils.pretty_format_dict(scores))

        bestscore_holder.compare_scores(scores["inkb_accuracy"]["accuracy"], 0)

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
        accum_count = 0

        progress_bar = tqdm(total=total_update_steps, desc="training steps")
        for epoch in range(1, max_epoch + 1):

            perm = np.random.permutation(n_train)

            # Negative Sampling
            # For each epoch, we generate candidate entities for each document
            # Note that candidate entities are generated per document
            # list[dict[str, list[CandEntKeyInfo]]]
            if not system.index_made:
                system.make_index()
            candidate_entities = self._generate_candidate_entities(
                system=system,
                documents=train_documents
            )

            for instance_i in range(0, n_train, batch_size):

                ##################
                # Forward
                ##################

                batch_i += 1

                # Initialize loss
                batch_loss = 0.0
                actual_batchsize = 0
                actual_total_mentions = 0

                for doc_i in train_doc_indices[
                    perm[instance_i: instance_i + batch_size]
                ]:
                    # Forward and compute loss
                    one_loss, n_valid_mentions = system.compute_loss(
                        document=train_documents[doc_i],
                        candidate_entities_for_doc=candidate_entities[doc_i]
                    )
                    # Accumulate the loss
                    batch_loss = batch_loss + one_loss
                    actual_batchsize += 1
                    actual_total_mentions += n_valid_mentions

                # Average the loss
                actual_batchsize = float(actual_batchsize)
                actual_total_mentions = float(actual_total_mentions)
                # loss per mention
                batch_loss = batch_loss / actual_total_mentions

                ##################
                # Backward
                ##################

                batch_loss = batch_loss / gradient_accumulation_steps
                batch_loss.backward()
                # with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()

                # Accumulate for reporting
                loss_accum += float(batch_loss.cpu())
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
                        "max_valid_inkb_acc": bestscore_holder.best_score,
                        "patience": bestscore_holder.patience
                    }
                    writer_train.write(out)
                    logger.info(utils.pretty_format_dict(out))
                    loss_accum = 0.0
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

                    system.make_index()

                    scores = self.evaluate(
                        system=system,
                        documents=dev_documents,
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
                    logger.info("[Step %d] Max validation InKB accuracy: %f" % (step, bestscore_holder.best_score))

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
        split,
        #
        prediction_only=False,
        get_scores_only=False,
    ):
        """
        Parameters
        ----------
        system : EntityRetrievalBiEncoderSystem
        documents : list[Document]
        split : str
        prediction_only : bool
            by default False
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any] | None
        """
        # (documents, entity_dict) -> path_pred
        result_documents, candidate_entities = system.batch_extract(
            documents=documents,
            retrieval_size=system.config["retrieval_size"]
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        utils.write_json(
            self.paths[f"path_{split}_pred_retrieval"],
            candidate_entities
        )
        if prediction_only:
            return
        # (path_pred, path_gold) -> scores
        scores = evaluation.ed.accuracy(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True,
            skip_normalization=True
        )
        scores.update(evaluation.ed.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True,
            skip_normalization=True
        ))
        scores.update(evaluation.ed.recall_at_k(
            pred_path=self.paths[f"path_{split}_pred_retrieval"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        ))
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

    def _generate_candidate_entities(self, system, documents):
        """
        Parameters
        ----------
        system : EntityRetrievalBiEncoderSystem
        documents : list[Document]

        Returns
        -------
        list[dict[str, list[CandEntKeyInfo]]]
        """
        RETRIEVAL_SIZE = 10 # the number of retrieved entities for each mention

        candidate_entities = [] # list[dict[str, list[CandEntKeyInfo]]]

        all_entity_ids = list(system.entity_dict.keys())
        n_hard_negatives = 0
        n_random_negatives = 0
        for document in tqdm(documents, desc="candidate generation"):
            candidate_entities_for_doc = [] # list[CandEntKeyInfo]

            # Get gold entities
            gold_entity_ids = list(set([
                m["entity_id"] for m in document["mentions"]
            ])) # list[str]
            assert len(gold_entity_ids) < system.config["n_candidate_entities"]
            gold_entities = [
                {
                    "entity_id": eid,
                    "score": float("inf"),
                }
                for eid in gold_entity_ids
            ] # list[CandEntKeyInfo]

            memorized_entity_ids = set(gold_entity_ids)

            # Retrieve entities for each mention
            # list[list[CandEntKeyInfo]]
            pred_document, candidate_entities_for_doc \
                = system.extract(
                    document=document,
                    retrieval_size=RETRIEVAL_SIZE
                )
            candidate_entities_for_mentions \
                = candidate_entities_for_doc["candidate_entities"]

            # Get retrieval-based (hard-negative or non-hard-negative) entities
            # hard negatives: entities whose scores are greater than
            #   the retrieval score for the gold entity.
            # ---
            hard_negative_entities = [] # list[CandEntKeyInfo]
            non_hard_negative_entities = [] # list[CandEntKeyInfo]
            for m_i in range(len(document["mentions"])):
                # Identify the retrieval score for the gold entity
                gold_entity_id = document["mentions"][m_i]["entity_id"]
                tmp = [
                    cand["score"]
                    for cand in candidate_entities_for_mentions[m_i]
                    if cand["entity_id"] == gold_entity_id
                ]
                if len(tmp) == 0:
                    # No gold entity found in the candidates
                    gold_score = -1
                elif len(tmp) == 1:
                    gold_score = tmp[0]
                else:
                    raise Exception(f"Gold entity ({gold_entity_id}) is found multiple times in the retrieved results")
                # Identify hard-negative or non-hard-negative entities
                hard_negative_entities.extend([
                    cand for cand in candidate_entities_for_mentions[m_i]
                    if cand["score"] >= gold_score
                ])
                non_hard_negative_entities.extend([
                    cand for cand in candidate_entities_for_mentions[m_i]
                    if cand["score"] < gold_score
                ])

            # Sort the retrieval-based entities based on their retrieval
            #   scores in descending order.
            hard_negative_entities = sorted(
                hard_negative_entities,
                key=lambda cand: -cand["score"]
            )
            non_hard_negative_entities = sorted(
                non_hard_negative_entities,
                key=lambda cand: -cand["score"]
            )
            # Remove duplicate retrieval-based entities
            tmp_hard_negative_entities = []
            for cand in hard_negative_entities:
                if not cand["entity_id"] in memorized_entity_ids:
                    tmp_hard_negative_entities.append(cand)
                    memorized_entity_ids.add(cand["entity_id"])
            hard_negative_entities = tmp_hard_negative_entities
            tmp_non_hard_negative_entities = []
            for cand in non_hard_negative_entities:
                if not cand["entity_id"] in memorized_entity_ids:
                    tmp_non_hard_negative_entities.append(cand)
                    memorized_entity_ids.add(cand["entity_id"])
            non_hard_negative_entities = tmp_non_hard_negative_entities

            n_hard_negatives += len(hard_negative_entities)

            # Combine the gold and retrieval-based entities
            candidate_entities_for_doc = (
                gold_entities
                + hard_negative_entities
                + non_hard_negative_entities
            )

            # Adjust the number of candidates to the specified number
            candidate_entities_for_doc \
                = candidate_entities_for_doc[
                    :system.config["n_candidate_entities"]
                ]

            # Sample entities randomly if the number of candidates is
            #   less than the specified number
            while (
                len(candidate_entities_for_doc)
                < system.config["n_candidate_entities"]
            ):
                eid = np.random.choice(all_entity_ids)
                if not eid in memorized_entity_ids:
                    candidate_entities_for_doc.append({
                        "entity_id": eid,
                        "score": 0.0,
                    })
                    memorized_entity_ids.add(eid)
                    n_random_negatives += 1

            # dict[str, list[CandEntKeyInfo]]
            candidate_entities_for_doc = {
                "candidate_entities": candidate_entities_for_doc
            }

            candidate_entities.append(candidate_entities_for_doc)

        logger.info(f"Avg. hard negatives (per doc): {float(n_hard_negatives) / len(documents)}")
        logger.info(f"Avg. random negatives (per doc): {float(n_random_negatives) / len(documents)}")

        return candidate_entities

