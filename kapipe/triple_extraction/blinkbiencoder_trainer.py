import copy
import logging
import os
import random
import time

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


class BlinkBiEncoderTrainer:

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

    def setup_dataset(self, extractor, documents, split):
        """
        Parameters
        ----------
        extractor : BlinkBiEncoder
        documents : list[Document]
        split : str
        """
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(extractor.entity_dict.keys()))
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
        extractor,
        train_documents,
        dev_documents,
    ):
        """
        Parameters
        ----------
        extractor : BlinkBiEncoder
        train_documents : list[Document]
        dev_documents : list[Document]
        """
        train_doc_indices = np.arange(len(train_documents))

        ##################
        # Get optimizer and scheduler
        ##################

        n_train = len(train_doc_indices)
        max_epoch = extractor.config["max_epoch"]
        batch_size = extractor.config["batch_size"]
        gradient_accumulation_steps \
            = extractor.config["gradient_accumulation_steps"]
        total_update_steps \
            = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * extractor.config["warmup_ratio"])

        logger.info("Number of training documents: %d" % n_train)
        logger.info("Number of epochs: %d" % max_epoch)
        logger.info("Batch size: %d" % batch_size)
        logger.info("Gradient accumulation steps: %d" % gradient_accumulation_steps)
        logger.info("Total update steps: %d" % total_update_steps)
        logger.info("Warmup steps: %d" % warmup_steps)

        optimizer = shared_functions.get_optimizer2(
            model=extractor.model,
            config=extractor.config
        )
        # extractor.model, optimizer = amp.initialize(
        #     extractor.model,
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

        # Perform indexing before evaluation
        extractor.make_index()

        scores = self.evaluate(
            extractor=extractor,
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

        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], extractor.config)
        utils.write_json(self.paths["path_config"], extractor.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

        # Save the model
        extractor.save_model(path=self.paths["path_snapshot"])
        logger.info("Saved model to %s" % self.paths["path_snapshot"])

        ##################
        # Training-and-validation loops
        ##################

        bert_param, task_param = extractor.model.get_params()
        extractor.model.zero_grad()
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
            # if not extractor.index_made:
            #     extractor.make_index()
            flatten_candidate_entities = self._generate_flatten_candidate_entities(
                extractor=extractor,
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
                    one_loss, n_valid_mentions = extractor.compute_loss(
                        document=train_documents[doc_i],
                        flatten_candidate_entities_for_doc=flatten_candidate_entities[doc_i]
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

                    if extractor.config["max_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            bert_param,
                            extractor.config["max_grad_norm"]
                        )
                        torch.nn.utils.clip_grad_norm_(
                            task_param,
                            extractor.config["max_grad_norm"]
                        )
                        # torch.nn.utils.clip_grad_norm_(
                        #     amp.master_params(optimizer),
                        #     extractor.config["max_grad_norm"]
                        # )
                    optimizer.step()
                    scheduler.step()

                    extractor.model.zero_grad()
                    step += 1
                    progress_bar.update()
                    progress_bar.refresh()

                if (
                    (instance_i + batch_size >= n_train)
                    or
                    (
                        (batch_i % gradient_accumulation_steps == 0)
                        and
                        (step % extractor.config["n_steps_for_monitoring"] == 0)
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
                        (extractor.config["n_steps_for_validation"] > 0)
                        and
                        (step % extractor.config["n_steps_for_validation"] == 0)
                    )
                ):

                    ##################
                    # Evaluate
                    ##################

                    # Perform indexing before evaluation
                    extractor.make_index()

                    scores = self.evaluate(
                        extractor=extractor,
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
                        extractor.save_model(path=self.paths["path_snapshot"])
                        logger.info("Saved model to %s" % self.paths["path_snapshot"])

                    if (
                        bestscore_holder.patience
                        >= extractor.config["max_patience"]
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
        extractor,
        documents,
        split,
        #
        prediction_only=False,
        get_scores_only=False,
    ):
        """
        Parameters
        ----------
        extractor : BlinkBiEncoder
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
        result_documents, candidate_entities = extractor.batch_extract(
            documents=documents,
            retrieval_size=extractor.config["retrieval_size"]
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

    def _generate_flatten_candidate_entities(self, extractor, documents):
        """
        Parameters
        ----------
        extractor : BlinkBiEncoder
        documents : list[Document]

        Returns
        -------
        list[dict[str, list[CandEntKeyInfo]]]
        """
        RETRIEVAL_SIZE = 10 # the number of retrieved entities for each mention

        logger.info("Generating candidate entities for training ...")
        start_time = time.time()

        flatten_candidate_entities = [] # list[dict[str, list[CandEntKeyInfo]]]

        # Predict candidate entities for each mention in each document
        _, candidate_entities = extractor.batch_extract(
            documents=documents,
            retrieval_size=RETRIEVAL_SIZE
        )
 
        all_entity_ids = set(list(extractor.entity_dict.keys()))
        n_total_mentions = 0
        n_inbatch_negatives = 0
        n_hard_negatives = 0
        n_nonhard_negatives = 0

        for document, candidate_entities_for_doc in tqdm(
            zip(documents, candidate_entities),
            total=len(documents),
            desc="candidate generation"
        ):
            #############
            # Gold entities
            #############

            # Aggregate gold entities for the mentions in the document
            gold_entity_ids = list(set([
                m["entity_id"] for m in document["mentions"]
            ])) # list[str]
            assert len(gold_entity_ids) <= extractor.config["n_candidate_entities"]

            tuples = [(eid, 0, float("inf")) for eid in gold_entity_ids]

            n_mentions = len(document["mentions"])
            n_total_mentions += n_mentions
            n_inbatch_negatives += (len(gold_entity_ids) - 1) * n_mentions

            #############
            # Hard-negative and non-hard-negative entities
            # Hard Negatives: entities whose scores are greater than the retrieval score for the gold entity
            #############

            # Aggregate hard-negative and non-hard-negative entities for the mentions in the document
            for mention, candidate_entities_for_mention in zip(
                document["mentions"],
                candidate_entities_for_doc["candidate_entities"]
            ):
                # Identify the retrieval score of the gold entity for the mention
                gold_entity_id = mention["entity_id"]
                gold_score = next(
                    (
                        c["score"] for c in candidate_entities_for_mention
                        if c["entity_id"] == gold_entity_id
                    ),
                    -1.0
                )

                # Split the retrieved entities into hard negatives and non-hard negatives
                hard_negative_tuples = [
                    (c["entity_id"], 1, c["score"])
                    for c in candidate_entities_for_mention
                    if c["score"] >= gold_score and c["entity_id"] != gold_entity_id
                ]
                non_hard_negative_tuples = [
                    (c["entity_id"], 2, c["score"])
                    for c in candidate_entities_for_mention
                    if c["score"] < gold_score
                ]

                n_hard_negatives += len(hard_negative_tuples)
                n_nonhard_negatives += len(non_hard_negative_tuples)

                tuples.extend(hard_negative_tuples + non_hard_negative_tuples)

            #############
            # Combine and select candidate entities from the gold, hard-negative, and non-hard-negative entities
            #############

            # Sort the entities based on the types (gold/hard-negative/non-hard-negative) and then scores
            tuples = sorted(tuples, key=lambda x: (x[1], -x[2]))

            # Remove duplicate entities
            id_to_score = {}
            for eid, _, score in tuples:
                if not eid in id_to_score:
                    id_to_score[eid] = score
            tuples = list(id_to_score.items())

            tuples = tuples[:extractor.config["n_candidate_entities"]]

            #############
            # Sample entities randomly if the number of candidates is less than the specified number
            #############

            N = extractor.config["n_candidate_entities"]
            M = len(tuples)

            if N - M > 0:
                # Identify entities that are not contained in the current candidates
                possible_entity_ids = list(
                    all_entity_ids - set([eid for (eid,score) in tuples])
                )

                # Perform random sampling to get additinal candidate entities
                additional_entity_ids = random.sample(possible_entity_ids, N - M)
                additional_tuples = [
                    (eid, 0.0) for eid in additional_entity_ids
                ]

                tuples.extend(additional_tuples)

            #############
            # Formatting
            #############

            flatten_candidate_entities_for_doc = [
                {
                    "entity_id": eid,
                    "score": score
                }
                for (eid, score) in tuples
            ]

            # dict[str, list[CandEntKeyInfo]]
            flatten_candidate_entities_for_doc = {
                "flatten_candidate_entities": flatten_candidate_entities_for_doc
            }

            flatten_candidate_entities.append(flatten_candidate_entities_for_doc)

        end_time = time.time()
        span_time = end_time - start_time
        span_time /= 60.0

        logger.info(f"Avg. in-batch negatives (per mention): {float(n_inbatch_negatives) / n_total_mentions}")
        logger.info(f"Avg. hard negatives (per mention): {float(n_hard_negatives) / n_total_mentions}")
        logger.info(f"Avg. non-hard negatives (per mention): {float(n_nonhard_negatives) / n_total_mentions}")
        logger.info(f"Time: {span_time} min.")

        return flatten_candidate_entities

