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


class MAQATrainer:

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
        Dict[str, str]
        """
        paths = {}

        # Path to config file
        paths["path_config"] = self.base_output_path + "/config"
        # Path to vocabulary
        paths["path_vocab_answer"] = self.base_output_path + "/answers.vocab.txt"
        # Path to model snapshot
        paths["path_snapshot"] = self.base_output_path + "/model"

        # Path to gold training triples for Ign evaluation
        paths["path_gold_train_triples"] = self.base_output_path + "/gold_train_triples.json"

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

    def setup_dataset(
        self,
        system,
        documents,
        split,
        with_gold_annotations=True
    ):
        """
        Parameters
        ----------
        system : MAQASystem
        documents : list[Document]
        split : str
        with_gold_annotations : bool
            by default True
        """
        # Cache the gold training triples for Ign evaluation
        if split == "train":
            if not os.path.exists(self.paths["path_gold_train_triples"]):
                gold_train_triples = []
                for document in tqdm(documents, desc="dataset setup"):
                    mentions = document["mentions"]
                    entity_index_to_mention_names = {
                        e_i: [
                            mentions[m_i]["name"]
                            for m_i in e["mention_indices"]
                        ]
                        for e_i, e in enumerate(document["entities"])
                    }
                    for triple in document["relations"]:
                        arg1_entity_i = triple["arg1"]
                        rel = triple["relation"]
                        arg2_entity_i = triple["arg2"]
                        arg1_mention_names \
                                = entity_index_to_mention_names[arg1_entity_i]
                        arg2_mention_names \
                                = entity_index_to_mention_names[arg2_entity_i]
                        for arg1_mention_name in arg1_mention_names:
                            for arg2_mention_name in arg2_mention_names:
                                gold_train_triples.append((
                                    arg1_mention_name,
                                    rel,
                                    arg2_mention_name
                                ))
                gold_train_triples = list(set(gold_train_triples))
                gold_train_triples = {"root": gold_train_triples}
                utils.write_json(
                    self.paths["path_gold_train_triples"],
                    gold_train_triples
                )
                logger.info(f"Saved the gold training triples for Ign evaluation in {self.paths['path_gold_train_triples']}")

        # Cache the gold annotations for evaluation
        if split !=  "train" and with_gold_annotations:
            path_gold = self.paths[f"path_{split}_gold"]
            if not os.path.exists(path_gold):
                gold_documents = []
                for document in tqdm(documents, desc="dataset setup"):
                    gold_doc = copy.deepcopy(document)
                    gold_documents.append(gold_doc)
                utils.write_json(path_gold, gold_documents)
                logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def train(
        self,
        system,
        train_documents,
        dev_documents,
        supplemental_info
    ):
        """
        Parameters
        ----------
        system : MAQASystem
        train_documents : list[Document]
        dev_documents : list[Document]
        supplemental_info: dict[str, Any]
        """
        train_doc_indices = np.arange(len(train_documents))

        # We expand the training documents for each QA level,
        #   because each document consists of the different number of QAs.
        # First, aggregate (doc_i, qa_index) tuples
        #   for positive and negative questions separately.
        pos_train_tuples = [] # list[tuple[int, int]]
        neg_train_tuples = [] # list[tuple[int, int]]
        for doc_i in train_doc_indices:
            document = train_documents[doc_i]
            preprocessed_data = system.preprocessor.preprocess(
                document=document
            )
            qas = preprocessed_data["qas"]
            for qa_i in range(len(qas)):
                answer = qas[qa_i].answer # str
                if answer == "Yes":
                    pos_train_tuples.append((doc_i, qa_i))
                elif answer == "No":
                    neg_train_tuples.append((doc_i, qa_i))
                else:
                    raise Exception(f"Invalid answer: {answer}")
        n_pos_train = len(pos_train_tuples)
        n_neg_train_before_sampling = len(neg_train_tuples)
        # Then, perform negative-question sampling
        if system.config["n_negative_samples"] > 0:
            perm = np.random.permutation(len(neg_train_tuples))
            perm = perm[
                0 : len(pos_train_tuples) * system.config["n_negative_samples"]
            ]
            neg_train_tuples = [neg_train_tuples[i] for i in perm]
        n_neg_train_after_sampling = len(neg_train_tuples)
        # Finally, concatenate the positive and negative tuples
        train_doc_index_and_qa_index_tuples = pos_train_tuples + neg_train_tuples
        train_doc_index_and_qa_index_tuples = np.asarray(
            train_doc_index_and_qa_index_tuples
        )

        ##################
        # Get optimizer and scheduler
        ##################

        n_train = len(train_doc_index_and_qa_index_tuples)
        max_epoch = system.config["max_epoch"]
        batch_size = system.config["batch_size"]
        gradient_accumulation_steps \
            = system.config["gradient_accumulation_steps"]
        total_update_steps \
            = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * system.config["warmup_ratio"])

        logger.info(f"Number of training QAs (all): {n_pos_train} (pos) + {n_neg_train_before_sampling} (neg) = {n_pos_train + n_neg_train_before_sampling}")
        logger.info(f"Number of training QAs (after negative sampling): {n_pos_train} (pos) + {n_neg_train_after_sampling} (neg) = {n_pos_train + n_neg_train_after_sampling}")
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

        if system.config["use_official_evaluation"]:
            scores = self.official_evaluate(
                system=system,
                documents=dev_documents,
                split="dev",
                supplemental_info=supplemental_info,
                #
                get_scores_only=True
            )
        else:
            scores = self.evaluate(
                system=system,
                documents=dev_documents,
                split="dev",
                supplemental_info=supplemental_info,
                #
                skip_intra_inter=True,
                skip_ign=True,
                get_scores_only=True
            )
        scores["epoch"] = 0
        scores["step"] = 0
        writer_dev.write(scores)
        logger.info(utils.pretty_format_dict(scores))

        bestscore_holder.compare_scores(scores["standard"]["f1"], 0)

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

        # Save the vocabulary (only once)
        utils.write_vocab(
            self.paths["path_vocab_answer"],
            system.vocab_answer,
            write_frequency=False
        )
        logger.info("Saved answer type vocabulary to %s" % self.paths["path_vocab_answer"])

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

                for (doc_i, qa_i) in train_doc_index_and_qa_index_tuples[
                    perm[instance_i: instance_i + batch_size]
                ]:
                    doc_i = int(doc_i)
                    qa_i = int(qa_i)

                    # Forward and compute loss
                    one_loss, one_acc = system.compute_loss(
                        document=train_documents[doc_i],
                        qa_index=qa_i
                    )

                    # Accumulate the loss
                    batch_loss = batch_loss + one_loss
                    batch_acc = batch_acc + one_acc
                    actual_batchsize += 1

                # Average the loss
                actual_batchsize = float(actual_batchsize)
                batch_loss = batch_loss / actual_batchsize # loss per pair
                batch_acc = batch_acc / actual_batchsize

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
                        "max_valid_f1": bestscore_holder.best_score,
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

                    if system.config["use_official_evaluation"]:
                        scores = self.official_evaluate(
                            system=system,
                            documents=dev_documents,
                            split="dev",
                            supplemental_info=supplemental_info,
                            #
                            get_scores_only=True
                        )
                    else:
                        scores = self.evaluate(
                            system=system,
                            documents=dev_documents,
                            split="dev",
                            supplemental_info=supplemental_info,
                            #
                            skip_intra_inter=True,
                            skip_ign=True,
                            get_scores_only=True
                        )
                    scores["epoch"] = epoch
                    scores["step"] = step
                    writer_dev.write(scores)
                    logger.info(utils.pretty_format_dict(scores))

                    did_update = bestscore_holder.compare_scores(
                        scores["standard"]["f1"],
                        epoch
                    )
                    logger.info("[Step %d] Max validation F1: %f" % (step, bestscore_holder.best_score))

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
        supplemental_info,
        #
        skip_intra_inter=False,
        skip_ign=False,
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        system : MAQASystem
        documents : list[Document]
        split : str
        supplemental_info : dict[str, Any]
        skip_intra_inter : bool
            by default False
        skip_ign : bool
            by default False
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # documents -> path_pred
        result_documents = system.batch_extract(documents=documents)
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        # (path_pred, path_gold) -> scores
        scores = evaluation.docre.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            skip_intra_inter=skip_intra_inter,
            skip_ign=skip_ign,
            gold_train_triples_path=self.paths["path_gold_train_triples"]
        )
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

    def official_evaluate(
        self,
        system,
        documents,
        split,
        supplemental_info,
        #
        prediction_only=False,
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        system : MAQASystem
        documents : list[Document]
        split : str
        supplemental_info : dict[str, Any]
        prediction_only : bool
            by default False
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # NOTE: prediction_only is assumed to be used for the DocRED test set
        # documents -> path_pred
        result_documents = system.batch_extract(documents=documents)
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        # path_pred -> triples (path_pred variant)
        triples = evaluation.docre.to_official(
            path_input=self.paths[f"path_{split}_pred"],
            path_output=
            self.paths[f"path_{split}_pred"].replace(".json", ".official.json")
        )
        if prediction_only:
            return
        # gold info
        original_data_dir = supplemental_info["original_data_dir"]
        train_file_name = supplemental_info["train_file_name"]
        dev_file_name = supplemental_info[f"{split}_file_name"]
        # (triples, gold info) -> scores
        scores = evaluation.docre.official_evaluate(
            triples=triples,
            original_data_dir=original_data_dir,
            train_file_name=train_file_name,
            dev_file_name=dev_file_name
        )
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
