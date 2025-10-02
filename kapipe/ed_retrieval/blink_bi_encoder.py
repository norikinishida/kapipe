from __future__ import annotations

from collections import defaultdict
from collections import OrderedDict
import copy
import logging
import math
import os
import queue
import random
import time
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from opt_einsum import contract
import torch.multiprocessing as mp
from tqdm import tqdm
from tqdm.autonotebook import trange
import jsonlines

from ..nn_utils import get_optimizer2, get_scheduler2
from ..passage_retrieval import ApproximateNearestNeighborSearch
from .. import evaluation
from .. import utils
from ..utils import BestScoreHolder
from ..datatypes import (
    Config,
    Document,
    Mention,
    Entity,
    EntityPassage,
    CandEntKeyInfo,
    CandidateEntitiesForDocument
)


logger = logging.getLogger(__name__)


class BlinkBiEncoder:
    """
    BLINK Bi-Encoder (Wu et al., 2020).
    """

    def __init__(
        self,
        device: str,
        # Initialization
        config: Config | str | None = None,
        path_entity_dict: str | None = None,
        # Loading
        path_snapshot: str | None = None
    ):
        logger.info("########## BlinkBiEncoder Initialization Starts ##########")

        self.device = device
        self.path_snapshot = path_snapshot

        if path_snapshot is not None:
            assert config is None
            assert path_entity_dict is None
            config = path_snapshot + "/config"
            path_entity_dict = path_snapshot + "/entity_dict.json"
            path_model = path_snapshot + "/model"
            path_entity_vectors = path_snapshot + "/entity_vectors.npy"

        # Load the configuration
        if isinstance(config, str):
            config_path = config
            config = utils.get_hocon_config(config_path=config_path)
            logger.info(f"Loaded configuration from {config_path}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

       # Load the entity dictionary
        logger.info(f"Loading entity dictionary from {path_entity_dict}")
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        logger.info(f"Completed loading of entity dictionary with {len(self.entity_dict)} entities from {path_entity_dict}")

        # Initialize the model
        self.model_name = config["model_name"]
        if self.model_name == "blink_bi_encoder_model":
            self.model = BlinkBiEncoderModel(
                device=device,
                bert_pretrained_name_or_path=config["bert_pretrained_name_or_path"],
                max_seg_len=config["max_seg_len"],
                entity_seq_length=config["entity_seq_length"]
            )
        else:
            raise Exception(f"Invalid model_name: {self.model_name}")

        # Show parameter shapes
        # logger.info("Model parameters:")
        # for name, param in self.model.named_parameters():
        #     logger.info(f"{name}: {tuple(param.shape)}")

        # Load trained model parameters and entity vectors
        if path_snapshot is not None:
            self.model.load_state_dict(
                torch.load(path_model, map_location=torch.device("cpu")),
                strict=False
            )
            logger.info(f"Loaded model parameters from {path_model}")

            self.precomputed_entity_vectors = np.load(path_entity_vectors)
            logger.info(f"Loaded entity vectors from {path_entity_vectors}")

        self.model.to(self.model.device)

        # Initialize the module for Approximate Nearest Neighbor Search
        # It might be better to select a different GPU ID for indexing from the GPU ID of the BLINK model to avoid OOM error
        self.anns = ApproximateNearestNeighborSearch(gpu_id=0) # TODO: Allow GPU-ID selection

        logger.info("########## BlinkBiEncoder Initialization Ends ##########")

    def save(self, path_snapshot: str, model_only: bool = False) -> None:
        path_config = path_snapshot + "/config"
        path_entity_dict = path_snapshot + "/entity_dict.json"
        path_model = path_snapshot + "/model"
        path_entity_vectors = path_snapshot + "/entity_vectors.npy"
        if not model_only:
            utils.write_json(path_config, self.config)
            utils.write_json(path_entity_dict, list(self.entity_dict.values()))
        torch.save(self.model.state_dict(), path_model)
        np.save(path_entity_vectors, self.precomputed_entity_vectors)

    def compute_loss(
        self,
        document: Document,
        flatten_candidate_entities_for_doc: dict[str, list[CandEntKeyInfo]],
    ) -> tuple[torch.Tensor, int]:
        # Switch to training mode
        self.model.train()

        ###############
        # Entity Encoding
        ###############

        # Create entity passages
        candidate_entity_passages: list[EntityPassage] = []
        for cand in flatten_candidate_entities_for_doc["flatten_candidate_entities"]:
            entity_id = cand["entity_id"]
            epage = self.entity_dict[entity_id]
            canonical_name = epage["canonical_name"]
            # synonyms = epage["synonyms"]
            description = epage["description"]
            entity_passage: EntityPassage = {
                "title": canonical_name,
                "text": description,
                "entity_id": entity_id,
            }
            candidate_entity_passages.append(entity_passage)

        # Preprocess entities
        preprocessed_data_e = self.model.preprocess_entities(
            candidate_entity_passages=candidate_entity_passages
        )

        # Tensorize entities 
        model_input_e = self.model.tensorize_entities(
            preprocessed_data=preprocessed_data_e,
            compute_loss=True
        )

        # Encode entities
        # (n_candidates, hidden_dim)
        candidate_entity_vectors = self.model.encode_entities(**model_input_e)         

        ###############
        # Mention Encoding
        ###############

        # Preprocess mentions
        preprocessed_data_m = self.model.preprocess_mentions(document=document)

        # Tensorize mentions
        model_input_m = self.model.tensorize_mentions(
            preprocessed_data=preprocessed_data_m,
            compute_loss=True
        )

        # Encode mentions
        # (n_mentions, hidden_dim)
        mention_vectors = self.model.encode_mentions(**model_input_m)

        ###############
        # Scoring
        ###############

        # Preprocess for scoring
        preprocessed_data = self.model.preprocess_for_scoring(
            mentions=document["mentions"],
            candidate_entity_passages=candidate_entity_passages
        )

        # Tensorize for scoring
        model_input = self.model.tensorize_for_scoring(
            preprocessed_data=preprocessed_data,
            compute_loss=True
        )

        # Compute scores
        model_output = self.model.forward_for_scoring(
            mention_vectors=mention_vectors,
            candidate_entity_vectors=candidate_entity_vectors,
            **model_input
        )

        return (
            model_output.loss,
            model_output.n_mentions
        )

    def make_index(self, use_precomputed_entity_vectors: bool = False) -> None:
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()
            start_time = time.time()

            # Create entity passages
            logger.info(f"Building passages for {len(self.entity_dict)} entities ...")
            entity_passages = []
            for entity_id, epage in self.entity_dict.items():
                canonical_name = epage["canonical_name"]
                # synonyms = epage["synonyms"]
                description = epage["description"]
                entity_passage = {
                    "title": canonical_name,
                    "text": description,
                    "entity_id": entity_id,
                }
                entity_passages.append(entity_passage)

            # Preprocess, tensorize, and encode entities
            if use_precomputed_entity_vectors:
                entity_vectors = self.precomputed_entity_vectors
            else:
                logger.info(f"Encoding {len(entity_passages)} entities ...")
                pool = self.model.start_multi_process_pool()
                entity_vectors = self.model.encode_multi_process(entity_passages, pool)
                self.model.stop_multi_process_pool(pool)
                self.model.to(self.device)

            # Make ANNS index
            logger.info(f"Indexing {len(entity_vectors)} entities ...")
            self.anns.make_index(
                passage_vectors=entity_vectors,
                passage_metadatas=[
                    {
                        "title": p["title"],
                        "entity_id": p["entity_id"]
                    }
                    for p in entity_passages
                ]
            )

            self.precomputed_entity_vectors = entity_vectors

            end_time = time.time()
            span_time = end_time - start_time
            span_time /= 60.0
            logger.info("Completed indexing")
            logger.info(f"Time: {span_time} min.")

    def search(self, document: Document, retrieval_size: int = 1) -> tuple[
        Document, CandidateEntitiesForDocument
    ]:
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Skip prediction if no mention appears
            if len(document["mentions"]) == 0:
                result_document = copy.deepcopy(document)
                result_document["entities"] = []
                candidate_entities_for_doc = {
                   "doc_key": result_document["doc_key"],
                   "candidate_entities": []
                }
                return result_document, candidate_entities_for_doc

            # Preprocess mentions
            preprocessed_data_m = self.model.preprocess_mentions(document=document)

            # Tensorize mentions
            model_input_m = self.model.tensorize_mentions(
                preprocessed_data=preprocessed_data_m,
                compute_loss=False
            )

            # Encode mentions
            # (n_mentions, hidden_dim)
            mention_vectors = self.model.encode_mentions(**model_input_m)

            # Apply Approximate Nearest Neighbor Search
            #   (n_mentions, retrieval_size),
            #   (n_mentions, retrieval_size),
            #   (n_mentions, retrieval_size)
            (
                _,
                mention_pred_entity_metadatas,
                retrieval_scores
            ) = self.anns.search(
                query_vectors=mention_vectors.cpu().numpy(),
                top_k=retrieval_size
            )
            mention_pred_entity_names = [
                [y["title"] for y in ys]
                for ys in mention_pred_entity_metadatas
            ]
            mention_pred_entity_ids = [
                [y["entity_id"] for y in ys]
                for ys in mention_pred_entity_metadatas
            ]

            # Structurize (1)
            # Transform to mention-level entity IDs
            mentions: list[Mention] = []
            for m_i in range(len(preprocessed_data_m["mentions"])):
                mentions.append({"entity_id": mention_pred_entity_ids[m_i][0]})

            # Structurize (2)
            # Transform to entity-level entity IDs
            # i.e., aggregate mentions based on the entity IDs
            entities: list[Entity] = utils.aggregate_mentions_to_entities(
                document=document,
                mentions=mentions
            )

            # Structuriaze (3)
            # Transform to candidate entities for each mention
            candidate_entities_for_mentions: list[list[CandEntKeyInfo]] = []
            n_mentions = len(mention_pred_entity_ids)
            assert len(mention_pred_entity_ids[0]) == retrieval_size
            for m_i in range(n_mentions):
                lst_cand_ent: list[CandEntKeyInfo] = []
                for c_i in range(retrieval_size):
                    cand_ent = {
                        "entity_id": mention_pred_entity_ids[m_i][c_i],
                        "canonical_name": mention_pred_entity_names[m_i][c_i],
                        "score": float(retrieval_scores[m_i][c_i]),
                    }
                    lst_cand_ent.append(cand_ent)
                candidate_entities_for_mentions.append(lst_cand_ent)

            # Integrate
            result_document = copy.deepcopy(document)
            for m_i in range(len(result_document["mentions"])):
                result_document["mentions"][m_i].update(mentions[m_i])
            result_document["entities"] = entities
            candidate_entities_for_doc = {
                "doc_key": result_document["doc_key"],
                "candidate_entities": candidate_entities_for_mentions
            }
            return result_document, candidate_entities_for_doc

    def batch_search(
        self,
        documents: list[Document],
        retrieval_size: int = 1
    ) -> tuple[list[Document], list[CandidateEntitiesForDocument]]:
        result_documents: list[Document] = []
        candidate_entities: list[CandidateEntitiesForDocument] = []
        for document in tqdm(documents, desc="retrieval steps"):
            result_document, candidate_entities_for_doc = self.search(
                document=document,
                retrieval_size=retrieval_size
            )
            result_documents.append(result_document)
            candidate_entities.append(candidate_entities_for_doc)
        return result_documents, candidate_entities


class BlinkBiEncoderTrainer:
    """
    Trainer class for Blink Bi-Encoder retriever.
    Handles training loop, evaluation, model saving, and early stopping.
    """
    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str, str]:
        return {
            # configurations
            "path_snapshot": self.base_output_path,
            # training outputs
            "path_train_losses": f"{self.base_output_path}/train.losses.jsonl",
            "path_dev_evals": f"{self.base_output_path}/dev.eval.jsonl",
            # evaluation outputs
            "path_dev_gold": f"{self.base_output_path}/dev.gold.json",
            "path_dev_pred": f"{self.base_output_path}/dev.pred.json",
            "path_dev_pred_retrieval": f"{self.base_output_path}/dev.pred_candidate_entities.json",
            "path_dev_eval": f"{self.base_output_path}/dev.eval.json",
            "path_test_gold": f"{self.base_output_path}/test.gold.json",
            "path_test_pred": f"{self.base_output_path}/test.pred.json",
            "path_test_pred_retrieval": f"{self.base_output_path}/test.pred_candidate_entities.json",
            "path_test_eval": f"{self.base_output_path}/test.eval.json",
            # For the reranking-model training in the later stage, we need to annotate candidate entities also for the training set
            "path_train_pred": f"{self.base_output_path}/train.pred.json",
            "path_train_pred_retrieval": f"{self.base_output_path}/train.pred_candidate_entities.json",
        }

    def setup_dataset(
        self,
        retriever: BlinkBiEncoder,
        documents: list[Document],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            # Extract all concepts from the entity dictionary
            kb_entity_ids = set(list(retriever.entity_dict.keys()))
            gold_documents = []
            for document in tqdm(documents, desc="dataset setup"):
                gold_doc = copy.deepcopy(document)
                for m_i, mention in enumerate(document["mentions"]):
                    # Mark whether the gold entity is included in the entity dictionary (KB)
                    in_kb = mention["entity_id"] in kb_entity_ids
                    gold_doc["mentions"][m_i]["in_kb"] = in_kb
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def train(
        self,
        retriever: BlinkBiEncoder,
        train_documents: list[Document],
        dev_documents: list[Document],
    ) -> None:
        ##################
        # Setup
        ##################

        train_doc_indices = np.arange(len(train_documents))

        n_train = len(train_doc_indices)
        max_epoch = retriever.config["max_epoch"]
        batch_size = retriever.config["batch_size"]
        gradient_accumulation_steps = retriever.config["gradient_accumulation_steps"]
        total_update_steps = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * retriever.config["warmup_ratio"])

        logger.info("Number of training documents: %d" % n_train)
        logger.info("Number of epochs: %d" % max_epoch)
        logger.info("Batch size: %d" % batch_size)
        logger.info("Gradient accumulation steps: %d" % gradient_accumulation_steps)
        logger.info("Total update steps: %d" % total_update_steps)
        logger.info("Warmup steps: %d" % warmup_steps)

        optimizer = get_optimizer2(
            model=retriever.model,
            config=retriever.config
        )
        scheduler = get_scheduler2(
            optimizer=optimizer,
            total_update_steps=total_update_steps,
            warmup_steps=warmup_steps
        )

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
        # Initial Validation
        ##################

        # Build index
        retriever.make_index()

        # Evaluate the retriever
        scores = self.evaluate(
            retriever=retriever,
            documents=dev_documents,
            split="dev",
            #
            get_scores_only=True
        )
        scores.update({"epoch": 0, "step": 0})
        writer_dev.write(scores)
        logger.info(utils.pretty_format_dict(scores))

        # Set the best validation score
        bestscore_holder.compare_scores(scores["inkb_accuracy"]["accuracy"], 0)

        # Save
        retriever.save(path_snapshot=self.paths["path_snapshot"])
        logger.info(f"Saved config, entity dictionary, model, and entity vectors to {self.paths['path_snapshot']}")

        ##################
        # Training Loop
        ##################

        bert_param, task_param = retriever.model.get_params()
        retriever.model.zero_grad()
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
            # if not retriever.index_made:
            #     retriever.make_index()
            flatten_candidate_entities = self._generate_flatten_candidate_entities(
                retriever=retriever,
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

                for doc_i in train_doc_indices[perm[instance_i: instance_i + batch_size]]:
                    # Forward and compute loss
                    one_loss, n_valid_mentions = retriever.compute_loss(
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
                batch_loss = batch_loss / actual_total_mentions # loss per mention

                ##################
                # Backward
                ##################

                batch_loss = batch_loss / gradient_accumulation_steps
                batch_loss.backward()

                # Accumulate for reporting
                loss_accum += float(batch_loss.cpu())
                accum_count += 1

                if batch_i % gradient_accumulation_steps == 0:

                    ##################
                    # Update
                    ##################

                    if retriever.config["max_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            bert_param,
                            retriever.config["max_grad_norm"]
                        )
                        torch.nn.utils.clip_grad_norm_(
                            task_param,
                            retriever.config["max_grad_norm"]
                        )

                    optimizer.step()
                    scheduler.step()

                    retriever.model.zero_grad()

                    step += 1
                    progress_bar.update()
                    progress_bar.refresh()

                if (
                    (instance_i + batch_size >= n_train)
                    or
                    (
                        (batch_i % gradient_accumulation_steps == 0)
                        and
                        (step % retriever.config["n_steps_for_monitoring"] == 0)
                    )
                ):

                    ##################
                    # Report
                    ##################

                    report = {
                        "step": step,
                        "epoch": epoch,
                        "step_progress": "%d/%d" % (step, total_update_steps),
                        "step_progress(ratio)": float(step) / total_update_steps * 100.0,
                        "one_epoch_progress": "%d/%d" % (instance_i + actual_batchsize, n_train),
                        "one_epoch_progress(ratio)": float(instance_i + actual_batchsize) / n_train * 100.0,
                        "loss": loss_accum / accum_count,
                        "max_valid_inkb_acc": bestscore_holder.best_score,
                        "patience": bestscore_holder.patience
                    }
                    writer_train.write(report)
                    logger.info(utils.pretty_format_dict(report))
                    loss_accum = 0.0
                    accum_count = 0

                if (
                    (instance_i + batch_size >= n_train)
                    or
                    (
                        (batch_i % gradient_accumulation_steps == 0)
                        and
                        (retriever.config["n_steps_for_validation"] > 0)
                        and
                        (step % retriever.config["n_steps_for_validation"] == 0)
                    )
                ):

                    ##################
                    # Validation
                    ##################

                    # Build index
                    retriever.make_index()

                    # Evaluate the retriever
                    scores = self.evaluate(
                        retriever=retriever,
                        documents=dev_documents,
                        split="dev",
                        #
                        get_scores_only=True
                    )
                    scores.update({"epoch": epoch, "step": step})
                    writer_dev.write(scores)
                    logger.info(utils.pretty_format_dict(scores))

                    # Update the best validation score
                    did_update = bestscore_holder.compare_scores(
                        scores["inkb_accuracy"]["accuracy"],
                        epoch
                    )
                    logger.info("[Step %d] Max validation InKB accuracy: %f" % (step, bestscore_holder.best_score))

                    # Save the model
                    if did_update:
                        retriever.save(
                            path_snapshot=self.paths["path_snapshot"],
                            model_only=True
                        )
                        logger.info(f"Saved model and entity vectors to {self.paths['path_snapshot']}")

                    ##################
                    # Termination Check
                    ##################

                    if bestscore_holder.patience >= retriever.config["max_patience"]:
                        writer_train.close()
                        writer_dev.close()
                        progress_bar.close()
                        return

        writer_train.close()
        writer_dev.close()
        progress_bar.close()

    def evaluate(
        self,
        retriever: BlinkBiEncoder,
        documents: list[Document],
        split: str,
        #
        prediction_only: bool = False,
        get_scores_only: bool = False,
    ) -> dict[str, Any] | None:
        # Apply the retriever
        result_documents, candidate_entities = retriever.batch_search(
            documents=documents,
            retrieval_size=retriever.config["retrieval_size"]
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        utils.write_json(
            self.paths[f"path_{split}_pred_retrieval"],
            candidate_entities
        )

        if prediction_only:
            return

        # Calculate the evaluation scores
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

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

    def _generate_flatten_candidate_entities(
        self,
        retriever: BlinkBiEncoder,
        documents: list[Document]
    ) -> list[dict[str, list[CandEntKeyInfo]]]:
        logger.info("Generating candidate entities for training ...")
        start_time = time.time()

        RETRIEVAL_SIZE = 10 # the number of retrieved entities for each mention

        flatten_candidate_entities: list[dict[str, list[CandEntKeyInfo]]] = []

        # Predict candidate entities for each mention in each document
        _, candidate_entities = retriever.batch_search(
            documents=documents,
            retrieval_size=RETRIEVAL_SIZE
        )
 
        all_entity_ids = set(list(retriever.entity_dict.keys()))
        n_total_mentions = 0
        n_inbatch_negatives = 0
        n_hard_negatives = 0
        n_nonhard_negatives = 0

        for document, candidate_entities_for_doc in tqdm(
            zip(documents, candidate_entities),
            total=len(documents),
            desc="candidate generation"
        ):
            # Aggregate gold entities for the mentions in the document
            gold_entity_ids = list(set([m["entity_id"] for m in document["mentions"]]))
            assert len(gold_entity_ids) <= retriever.config["n_candidate_entities"]

            tuples = [(eid, 0, float("inf")) for eid in gold_entity_ids]

            n_mentions = len(document["mentions"])
            n_total_mentions += n_mentions
            n_inbatch_negatives += (len(gold_entity_ids) - 1) * n_mentions

            # Aggregate hard-negative and non-hard-negative entities for the mentions in the document
            # Hard Negatives = entities whose scores are greater than the retrieval score for the gold entity
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

            # Now, `tuples` contains the gold, hard-negative, and non-hard-negative entities

            # Sort the entities based on the types and then scores
            tuples = sorted(tuples, key=lambda x: (x[1], -x[2]))

            # Remove duplicate entities
            id_to_score = {}
            for eid, _, score in tuples:
                if not eid in id_to_score:
                    id_to_score[eid] = score
            tuples = list(id_to_score.items())

            # Select top-k entities
            tuples = tuples[:retriever.config["n_candidate_entities"]]

            # Sample entities randomly if the number of candidates is less than the specified number
            N = retriever.config["n_candidate_entities"]
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

            # Create an output object
            flatten_candidate_entities_for_doc = {
                "flatten_candidate_entities": [
                    {
                        "entity_id": eid,
                        "score": score
                    }
                    for (eid, score) in tuples
                ]
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


class MentionTuple(NamedTuple):
    span: tuple[int, int]
    name: str
    entity_type: str
    entity_id: str | None


class EntityTuple(NamedTuple):
    mention_indices: list[int]
    entity_type: str
    entity_id: str


class BlinkBiEncoderModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        entity_seq_length
    ):
        """
        Parameters
        ----------
        device : str
        bert_pretrained_name_or_path : str
        max_seg_len : int
        entity_seq_length : int
        """
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.device = device
        self.bert_pretrained_name_or_path = bert_pretrained_name_or_path
        self.max_seg_len = max_seg_len
        self.entity_seq_length = entity_seq_length

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert_m, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )
        self.bert_e, _ = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert_m.config.hidden_size

        self.linear = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        ######
        # Preprocessor
        ######

        self.preprocessor = BlinkBiEncoderPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            entity_seq_length=self.entity_seq_length
        )
        
        ######
        # Loss Function
        ######

        self.loss_function = nn.CrossEntropyLoss(reduction="none")

    def _initialize_bert_and_tokenizer(self, pretrained_model_name_or_path):
        """
        Parameters
        ----------
        pretrained_model_name_or_path : str

        Returns
        -------
        tuple[AutoModel, AutoTokenizer]
        """
        bert = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            return_dict=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=None
        )
        return bert, tokenizer

    ########################
    # For optimization
    ########################

    def get_params(self, named=False):
        """
        Parameters
        ----------
        named : bool
            by default False

        Returns
        -------
        tuple[list[tuple[str, Any]], list[tuple[str, Any]]]
        """
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    ################
    # Forward pass (entity encoding)
    ################

    def preprocess_entities(self, candidate_entity_passages):
        """
        Parameters
        ----------
        candidate_entity_passages: list[EntityPassage]

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess_entities(
            candidate_entity_passages=candidate_entity_passages
        )
 
    def tensorize_entities(self, preprocessed_data, compute_loss, target_device=None):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        compute_loss : bool
        target_device : str | None
            by default None

        Returns
        -------
        dict[str, Any]
        """
        # XXX
        if target_device is None:
            device = self.device
        else:
            device = target_device

        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (batch_size, entity_seq_length)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_id"],
            # device=self.device # XXX
            device=device # XXX
        )

        # (batch_size, entity_seq_length)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_mask"],
            # device=self.device # XXX
            device=device # XXX
        )

        return model_input

    def encode_entities(
        self,
        segments_id,
        segments_mask,
        compute_loss
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (batch_size, entity_seq_length)
        segments_mask : torch.Tensor
            shape of (batch_size, entity_seq_length)
        compute_loss : bool

        Returns
        -------
        torch.Tensor
            shape of (batch_size, hidden_dim)
        """
        # Encode tokens by BERT
        # (batch_size, entity_seq_length, hidden_dim)
        segments_vec = self.encode_tokens_for_entities(
            segments_id=segments_id,
            segments_mask=segments_mask
        )

        # Get [CLS] vectors
        # (batch_size, hidden_dim)
        entity_vectors = segments_vec[:, 0, :]

        return entity_vectors

    ################
    # Forward pass (mention encoding)
    ################

    def preprocess_mentions(self, document):
        """
        Parameters
        ----------
        document : Document
        candidate_entity_passages: list[EntityPassage] | None

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess_mentions(
            document=document,
        )

    def tensorize_mentions(self, preprocessed_data, compute_loss):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        compute_loss : bool

        Returns
        -------
        dict[str, Any]
        """
        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (n_segments, max_seg_len)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_id"],
            device=self.device
        )

        # (n_segments, max_seg_len)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_mask"],
            device=self.device
        )

        # (n_mentions,)
        model_input["mention_begin_token_indices"] = torch.tensor(
            preprocessed_data["mention_begin_token_indices"],
            device=self.device
        )

        # (n_mentions,)
        model_input["mention_end_token_indices"] = torch.tensor(
            preprocessed_data["mention_end_token_indices"],
            device=self.device
        )

        return model_input

    def encode_mentions(
        self,
        segments_id,
        segments_mask,
        mention_begin_token_indices,
        mention_end_token_indices,
        compute_loss
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        mention_begin_token_indices : torch.Tensor
            shape of (n_mentions,)
        mention_end_token_indices : torch.Tensor
            shape of (n_mentions,)
        compute_loss : bool

        Returns
        -------
        torch.Tensor
            shape of (n_mentions, hidden_dim)
        """
        # Encode tokens by BERT
        # (n_tokens, hidden_dim)
        token_vectors = self.encode_tokens_for_mentions(
            segments_id=segments_id,
            segments_mask=segments_mask
        )

        # Compute mention vectors
        # (n_mentions, hidden_dim)
        mention_vectors = self.compute_mention_vectors(
            token_vectors=token_vectors,
            mention_begin_token_indices=mention_begin_token_indices,
            mention_end_token_indices=mention_end_token_indices
        )
        return mention_vectors

    ################
    # Forward pass (scoring)
    ################

    def preprocess_for_scoring(self, mentions, candidate_entity_passages):
        """
        Parameters
        ----------
        mentions: list[Mention]
        candidate_entity_passages: list[EntityPassage] | None

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess_for_scoring(
            mentions=mentions,
            candidate_entity_passages=candidate_entity_passages
        )

    def tensorize_for_scoring(self, preprocessed_data, compute_loss):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        compute_loss : bool

        Returns
        -------
        dict[str, Any]
        """
        model_input = {}

        model_input["compute_loss"] = compute_loss

        # We assume that this and `forward` functions
        #   (for scoring forward pass) is called only in training,
        #   since we use Approximate Nearest Neighbor Search library
        #   instead of the manual inter-product computation and search.
        assert compute_loss == True

        # (n_mentions,)
        model_input["mention_gold_candidate_entity_indices"] = torch.tensor(
            preprocessed_data["mention_gold_candidate_entity_indices"],
            device=self.device
        ).to(torch.long)

        return model_input

    def forward_for_scoring(
        self,
        mention_vectors,
        candidate_entity_vectors,
        compute_loss,
        mention_gold_candidate_entity_indices=None
    ):
        """
        Parameters
        ----------
        mention_vectors : torch.Tensor
            shape of (n_mentions, hidden_dim)
        candidate_entity_vectors : torch.Tensor
            shape of (n_candidates, hidden_dim)
        compute_loss : bool
        mention_gold_candidate_entity_indices : torch.Tensor | None
            shape of (n_mentions,)

        Returns
        -------
        ModelOutput
        """
        # (n_mentions, n_candidates)
        logits = contract(
            "md,ed->me",
            mention_vectors.to(torch.float),
            candidate_entity_vectors
        )

        if not compute_loss:
            return ModelOutput(
                logits=logits
            )

        # Compute loss (summed over mentions)
        # (n_mentions,)
        loss = self.loss_function(
            logits,
            mention_gold_candidate_entity_indices
        )
        loss = loss.sum() # Scalar

        n_mentions = len(mention_gold_candidate_entity_indices)

        return ModelOutput(
            logits=logits,
            loss=loss,
            n_mentions=n_mentions
        )

    ################
    # Subfunctions
    ################

    def encode_tokens_for_entities(self, segments_id, segments_mask):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (batch_size, entity_seq_length)
        segments_mask : torch.Tensor
            shape of (batch_size, entity_seq_length)

        Returns
        -------
        torch.Tensor
            shape of (batch_size, entity_seq_length, hidden_dim)
        """
        bert_output = self.bert_e(
            input_ids=segments_id,
            attention_mask=segments_mask,
            output_attentions=False,
            output_hidden_states=False
        )
        # (batch_size, entity_seq_length, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]
        return segments_vec

    def encode_tokens_for_mentions(self, segments_id, segments_mask):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)

        Returns
        -------
        torch.Tensor
            shape of (n_tokens, hidden_dim)
        """
        bert_output = self.bert_m(
            input_ids=segments_id,
            attention_mask=segments_mask,
            output_attentions=False,
            output_hidden_states=False
        )
        # (n_segments, max_seg_len, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]

        # Transform `segments_vec` to token vectors
        # (n_segments, max_seg_len)
        segments_mask_bool = segments_mask.to(torch.bool)
        # (n_tokens, hidden_dim)
        token_vectors = segments_vec[segments_mask_bool]
        return token_vectors

    def compute_mention_vectors(
        self,
        token_vectors,
        mention_begin_token_indices,
        mention_end_token_indices
    ):
        """
        Parameters
        ----------
        token_vectors : torch.Tensor
            shape of (n_tokens, hidden_dim)
        mention_begin_token_indices : torch.Tensor
            shape of (n_mentions,)
        mention_end_token_indices : torch.Tensor
            shape of (n_mentions,)

        Returns
        -------
        torch.Tensor
            shape of (n_mentions, hidden_dim)
        """
        # (n_mentions, hidden_dim)
        mention_begin_token_vectors = token_vectors[mention_begin_token_indices]
        # (n_mentions, hidden_dim)
        mention_end_token_vectors = token_vectors[mention_end_token_indices]
        # (n_mentions, 2 * hidden_dim)
        mention_vectors = torch.cat(
            [mention_begin_token_vectors, mention_end_token_vectors],
            dim=1
        )
        # (n_mentions, hidden_dim)
        mention_vectors = self.linear(mention_vectors)
        return mention_vectors

    ################
    # For multi-processing
    ################

    def start_multi_process_pool(self):
        # Identify target devices
        # TODO: Allow GPU-IDs selection
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # With multi-GPU mode, we skip GPU-0 to avoid OOM error.
                # Here, we assume that GPU-0 is set for the BLINK model.
                target_devices = [f"cuda:{i}" for i in range(1, torch.cuda.device_count())]
            else:
                target_devices = ["cuda:0"]
        else:
            logger.info("CUDA is not available. Starting 4 CPU workers")
            target_devices = ["cpu"] * 4
        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        self.to("cpu")
        self.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in target_devices:
            p = ctx.Process(
                target=BlinkBiEncoderModel._encode_multi_process_worker,
                args=(device_id, self, input_queue, output_queue),
                daemon=True
            )
            p.start()
            processes.append(p)

        return {
            "input": input_queue,
            "output": output_queue,
            "processes": processes
        }

    @staticmethod
    def stop_multi_process_pool(pool):
        for p in pool["processes"]:
            p.terminate()
            
        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()

    def encode_multi_process(self, entity_passages, pool):
        CHUNK_SIZE = 256

        n_examples = len(entity_passages)
        n_processes = len(pool["processes"])

        chunk_size = min(math.ceil(n_examples / n_processes / 10), CHUNK_SIZE)

        logger.info(f"Chunk data into {math.ceil(n_examples / chunk_size)} packages of size {chunk_size}")

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for passage in entity_passages:
            chunk.append(passage)
            if len(chunk) >= chunk_size:
                input_queue.put(
                    [last_chunk_id, chunk]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put(
                [last_chunk_id, chunk]
            )
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Chunks", disable=False)],
            key=lambda x: x[0]
        )

        embeddings = np.concatenate([result[1] for result in results_list])
        # embeddings = torch.cat([result[1] for result in results_list], dim=0).numpy()
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(target_device, model, input_queue, results_queue):
        while True:
            try:
                chunk_id, chunk = input_queue.get()
                # print(target_device, chunk_id, len(passages))

                with torch.no_grad():
                    model.eval()

                    model.to(target_device)

                    # Preprocess entities
                    preprocessed_data_e = model.preprocess_entities(
                        candidate_entity_passages=chunk
                    )
    
                    # Tensorize entities
                    model_input_e = model.tensorize_entities(
                        preprocessed_data=preprocessed_data_e,
                        compute_loss=False,
                        target_device=target_device
                    )
    
                    # Encode entities
                    # (CHUNK_SIZE, hidden_dim)
                    embeddings = model.encode_entities(**model_input_e)
                    embeddings = embeddings.cpu().numpy()
                    # print(target_device, chunk_id, len(passages), embeddings.shape)

                results_queue.put([chunk_id, embeddings])

            except queue.Empty:
                break
 

class BlinkBiEncoderPreprocessor:

    def __init__(
        self,
        tokenizer,
        max_seg_len,
        entity_seq_length
    ):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
        max_seg_len : int
        entity_seq_length : int
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.entity_seq_length = entity_seq_length

        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token

    def preprocess_entities(self, candidate_entity_passages):
        """
        Parameters
        ----------
        candidate_entity_passages : list[EntityPassage]

        Returns
        -------
        dict[str, Any]
        """
        preprocessed_data = OrderedDict()

        #####
        # texts: list[list[str]]
        #   (n_candidates,)
        #####

        # (n_candidates,)
        texts = [
            (p["title"] + " : " + p["text"]).split()
            for p in candidate_entity_passages
        ]
        preprocessed_data["texts"] = texts

        #####
        # bert_input: dict[str, Any]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        #####
        inputs = self.tokenizer(
            [" ".join(t) for t in texts],
            # max_length=self.max_seg_len,
            max_length=self.entity_seq_length,
            padding=True,
            truncation=True,
            return_overflowing_tokens=False
        )
        bert_input = {}
        bert_input["segments_id"] = inputs["input_ids"]
        bert_input["segments"] = [
            self.tokenizer.convert_ids_to_tokens(seg)
            for seg in inputs["input_ids"]
        ]
        bert_input["segments_mask"] = inputs["attention_mask"]
        preprocessed_data["bert_input"] = bert_input

        return preprocessed_data

    def preprocess_mentions(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        dict[str, Any]
        """
        preprocessed_data = OrderedDict()

        #####
        # doc_key: str
        # sentences: list[list[str]]
        # mentions: list[MentionTuple]
        # entities: list[EntityTuple]
        #####

        preprocessed_data["doc_key"] = document["doc_key"]

        sentences = [s.split() for s in document["sentences"]]
        preprocessed_data["sentences"] = sentences

        if "entity_id" in document["mentions"][0]:
            with_supervision = True
        else:
            with_supervision = False
        if with_supervision:
            mentions = [
                MentionTuple(
                    tuple(m["span"]),
                    m["name"],
                    m["entity_type"],
                    m["entity_id"]
                )
                for m in document["mentions"]
            ]
        else:
            mentions = [
                MentionTuple(
                    tuple(m["span"]),
                    m["name"],
                    m["entity_type"],
                    None
                )
                for m in document["mentions"]
            ]
        preprocessed_data["mentions"] = mentions

        if with_supervision:
            entities = [
                EntityTuple(
                    e["mention_indices"],
                    e["entity_type"],
                    e["entity_id"]
                )
                for e in document["entities"]
            ]
            preprocessed_data["entities"] = entities

        #####
        # mention_index_to_sentence_index: list[int]
        # sentence_index_to_mention_indices: list[list[int]]
        # mention_index_to_entity_index: list[int]
        #####

        # Mention index to sentence index
        token_index_to_sent_index = [] # list[int]
        for sent_i, sent in enumerate(sentences):
            token_index_to_sent_index.extend([sent_i for _ in range(len(sent))])
        mention_index_to_sentence_index = [] # list[int]
        for mention in mentions:
            (begin_token_index, end_token_index) = mention.span
            sentence_index = token_index_to_sent_index[begin_token_index]
            assert token_index_to_sent_index[end_token_index] == sentence_index
            mention_index_to_sentence_index.append(sentence_index)
        preprocessed_data["mention_index_to_sentence_index"] \
            = mention_index_to_sentence_index

        # Sentence index to mention indices
        sentence_index_to_mention_indices = [None] * len(sentences)
        for mention_i, sent_i in enumerate(mention_index_to_sentence_index):
            if sentence_index_to_mention_indices[sent_i] is None:
                sentence_index_to_mention_indices[sent_i] = [mention_i]
            else:
                sentence_index_to_mention_indices[sent_i].append(mention_i)
        for sent_i in range(len(sentences)):
            if sentence_index_to_mention_indices[sent_i] is None:
                sentence_index_to_mention_indices[sent_i] = []
        preprocessed_data["sentence_index_to_mention_indices"] \
            = sentence_index_to_mention_indices

        if with_supervision:
            # Mention index to entity index
            # NOTE: Although a single mention may belong to multiple entities,
            #       we assign only one entity index to each mention
            mention_index_to_entity_index = [None] * len(mentions)
            for entity_i, entity in enumerate(entities):
                for mention_i in entity.mention_indices:
                    mention_index_to_entity_index[mention_i] = entity_i
            preprocessed_data["mention_index_to_entity_index"] \
                = mention_index_to_entity_index

        #####
        # bert_input: dict[str, Any]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        # subtoken_index_to_word_index: list[int]
        # word_index_to_subtoken_indices: list[ist[int]]
        # subtoken_index_to_sentence_index: list[int]
        # mention_begin_token_indices: list[int]
        # mention_end_token_indices: list[int]
        #####

        (
            segments,
            segments_id,
            segments_mask,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index,
            mention_begin_token_indices,
            mention_end_token_indices
        ) = self.tokenize_and_split(
            sentences=sentences,
            mentions=mentions
        )
        bert_input = {}
        bert_input["segments"] = segments
        bert_input["segments_id"] = segments_id
        bert_input["segments_mask"] = segments_mask
        bert_input["subtoken_index_to_word_index"] \
            = subtoken_index_to_word_index
        bert_input["word_index_to_subtoken_indices"] \
            = word_index_to_subtoken_indices
        bert_input["subtoken_index_to_sentence_index"] \
            = subtoken_index_to_sentence_index
        preprocessed_data["bert_input"] = bert_input

        preprocessed_data["mention_begin_token_indices"] = mention_begin_token_indices
        preprocessed_data["mention_end_token_indices"] = mention_end_token_indices

        #####
        # mention_gold_entity_ids: list[str]
        #   (n_mentions,)
        # mention_gold_candidate_entity_indices: list[int]
        #   (n_mention,)
        #####

        # if with_supervision:
        #     # Candidate entities should be determined beforehand for each document
        #     # list[str]; (n_candidates,)
        #     candidate_entity_ids = [
        #         p["entity_id"] for p in candidate_entity_passages
        #     ]
        #     preprocessed_data["candidate_entity_ids"] = candidate_entity_ids

        #     # list[str]; (n_mentions,)
        #     mention_gold_entity_ids = [
        #         mention.entity_id for mention in mentions
        #     ]
        #     preprocessed_data["mention_gold_entity_ids"] = mention_gold_entity_ids

        #     # (n_mentions, n_candidates)
        #     matched = (
        #         np.asarray(mention_gold_entity_ids)[:,None]
        #         == np.asarray(candidate_entity_ids)
        #     )
        #     n_mentions = len(mention_gold_entity_ids)
        #     assert matched.sum() == n_mentions
        #     # list[int]; (n_mentions,)
        #     mention_gold_candidate_entity_indices = np.where(matched)[1].tolist()
        #     preprocessed_data["mention_gold_candidate_entity_indices"] \
        #         = mention_gold_candidate_entity_indices

        return preprocessed_data

    def preprocess_for_scoring(self, mentions, candidate_entity_passages):
        preprocessed_data = OrderedDict()

        #####
        # mention_gold_entity_ids: list[str]
        #   (n_mentions,)
        # mention_gold_candidate_entity_indices: list[int]
        #   (n_mention,)
        #####

        # Candidate entities should be determined beforehand for each document
        # list[str]; (n_candidates,)
        candidate_entity_ids = [
            p["entity_id"] for p in candidate_entity_passages
        ]
        preprocessed_data["candidate_entity_ids"] = candidate_entity_ids

        # list[str]; (n_mentions,)
        mention_gold_entity_ids = [
            # mention.entity_id for mention in mentions
            mention["entity_id"] for mention in mentions
        ]
        preprocessed_data["mention_gold_entity_ids"] = mention_gold_entity_ids

        # (n_mentions, n_candidates)
        matched = (
            np.asarray(mention_gold_entity_ids)[:,None]
            == np.asarray(candidate_entity_ids)
        )
        n_mentions = len(mention_gold_entity_ids)
        assert matched.sum() == n_mentions
        # list[int]; (n_mentions,)
        mention_gold_candidate_entity_indices = np.where(matched)[1].tolist()
        preprocessed_data["mention_gold_candidate_entity_indices"] \
            = mention_gold_candidate_entity_indices

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(self, sentences, mentions):
        """
        Parameters
        ----------
        sentences: list[list[str]]
        mentions: list[MentionTuple]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]], list[int],
            list[list[int]], list[int], list[int], list[int]]
        """
        # subtoken分割、subtoken単位でのtoken終点、文終点、メンション始点、
        #   メンション終点のindicatorsの作成
        (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index,
            sents_mention_begin,
            sents_mention_end
        ) = self.tokenize(sentences=sentences, mentions=mentions)

        # 1階層のリストに変換
        doc_subtoken = utils.flatten_lists(sents_subtoken) # list[str]
        doc_token_end = utils.flatten_lists(sents_token_end) # list[bool]
        doc_sentence_end = utils.flatten_lists(sents_sentence_end) # list[bool]
        doc_mention_begin = utils.flatten_lists(sents_mention_begin)
        doc_mention_end = utils.flatten_lists(sents_mention_end) # list[bool]

        assert sum(doc_mention_begin) == sum(doc_mention_end)

        # BERTセグメントに分割
        (
            segments,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index,
            mention_begin_token_indices,
            mention_end_token_indices
        ) = self.split(
            doc_subtoken=doc_subtoken,
            doc_sentence_end=doc_sentence_end,
            doc_token_end=doc_token_end,
            subtoken_index_to_word_index=subtoken_index_to_word_index,
            doc_mention_begin=doc_mention_begin,
            doc_mention_end=doc_mention_end
        )

        assert (
            len(mention_begin_token_indices)
            == len(mention_end_token_indices)
            == len(mentions)
        )

        # subtoken IDへの変換とpaddingマスクの作成
        segments_id, segments_mask = self.convert_to_token_ids_with_padding(
            segments=segments
        )

        return (
            segments,
            segments_id,
            segments_mask,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index,
            mention_begin_token_indices,
            mention_end_token_indices
        )

    def tokenize(self, sentences, mentions):
        """
        Parameters
        ----------
        sentences: list[list[str]]
        mentions: list[MentionTuple]

        Returns
        -------
        tuple[list[list[str]], list[list[bool]], list[list[bool]], list[int],
            list[list[bool]], list[list[bool]]]
        """
        sents_subtoken = [] # list[list[str]]
        sents_token_end = [] # list[list[bool]]
        sents_sentence_end = [] # list[list[bool]]
        subtoken_index_to_word_index = [] # list[int]

        sents_mention_begin = [] # list[list[bool]]
        sents_mention_end = [] # list[list[bool]]

        original_mention_begin_token_indices = np.asarray(
            [m.span[0] for m in mentions] + [-1]
        )
        original_mention_end_token_indices = np.asarray(
            [m.span[1] for m in mentions] + [-1]
        )

        word_idx = -1
        offset = 0
        for sent in sentences:
            sent_subtoken = [] # list[str]
            sent_token_end = [] # list[bool]
            sent_sentence_end = [] # list[bool]
            sent_mention_begin = [] # list[bool]
            sent_mention_end = [] # list[bool]
            for token_i, token in enumerate(sent):
                word_idx += 1
                # サブトークン
                subtokens = self.tokenizer.tokenize(token)
                if len(subtokens) == 0:
                    subtokens = [self.tokenizer.unk_token]
                sent_subtoken.extend(subtokens)
                # トークンの終了位置
                sent_token_end += [False] * (len(subtokens) - 1) + [True]
                # 文の終了位置 (仮)
                sent_sentence_end += [False] * len(subtokens)
                # subtoken index -> word index
                subtoken_index_to_word_index += [word_idx] * len(subtokens)
                # メンションの開始位置、終了位置
                begin_count = len(np.where(
                    original_mention_begin_token_indices == (offset + token_i)
                )[0])
                end_count = len(np.where(
                    original_mention_end_token_indices == (offset + token_i)
                )[0])
                sent_mention_begin += [begin_count] + [0] * (len(subtokens) - 1)
                sent_mention_end += [end_count] + [0] * (len(subtokens) - 1)
            # 文の終了位置
            sent_sentence_end[-1] = True
            assert sum(sent_mention_begin) == sum(sent_mention_end)
            sents_subtoken.append(sent_subtoken)
            sents_token_end.append(sent_token_end)
            sents_sentence_end.append(sent_sentence_end)
            sents_mention_begin.append(sent_mention_begin)
            sents_mention_end.append(sent_mention_end)

            offset += len(sent)

        return (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index,
            sents_mention_begin,
            sents_mention_end
        )

    def split(
        self,
        doc_subtoken,
        doc_sentence_end,
        doc_token_end,
        subtoken_index_to_word_index,
        doc_mention_begin,
        doc_mention_end
    ):
        """
        Parameters
        ----------
        doc_subtoken: list[str]
        doc_sentence_end: list[bool]
        doc_token_end: list[bool]
        subtoken_index_to_word_index: list[int]
        doc_mention_begin: list[bool]
        doc_mention_end: list[bool]

        Returns
        -------
        tuple[list[list[str]], list[int], list[list[int]], list[int],
            list[int], list[int]]
        """
        segments = [] # list[list[str]]
        segments_subtoken_map = [] # list[list[int]]
        segments_mention_begin = [] # list[list[bool]]
        segments_mention_end = [] # list[list[bool]]

        n_subtokens = len(doc_subtoken)
        curr_idx = 0 # Index for subtokens
        while curr_idx < len(doc_subtoken):
            # Try to split at a sentence end point
            end_idx = min(curr_idx + self.max_seg_len - 1 - 2, n_subtokens - 1)
            while end_idx >= curr_idx and not doc_sentence_end[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.warning("No sentence end found; split at token end")
                # If no sentence end point found, try to split at token end point
                end_idx = min(
                    curr_idx + self.max_seg_len - 1 - 2,
                    n_subtokens - 1
                )
                seg_before = \
                    "\"" + " ".join(doc_subtoken[curr_idx:end_idx+1]) + "\""
                while end_idx >= curr_idx and not doc_token_end[end_idx]:
                    end_idx -= 1
                if end_idx < curr_idx:
                    logger.warning("Cannot split valid segment: no sentence end or token end")
                    raise Exception
                seg_after \
                    = "\"" + " ".join(doc_subtoken[curr_idx:end_idx+1]) + "\""
                logger.warning("------")
                logger.warning("Segment where no sentence-ending position was found:")
                logger.warning(seg_before)
                logger.warning("---")
                logger.warning("Segment splitted based on a token-ending position:")
                logger.warning(seg_after)
                logger.warning("------")

            segment = doc_subtoken[curr_idx: end_idx + 1]
            segment_subtoken_map = \
                subtoken_index_to_word_index[curr_idx: end_idx + 1]
            segment_mention_begin = doc_mention_begin[curr_idx: end_idx + 1]
            segment_mention_end = doc_mention_end[curr_idx: end_idx + 1]
            segment = [self.cls_token] + segment + [self.sep_token]
            # NOTE: [CLS] is treated as the first subtoken
            #       of the first word for each segment
            # NOTE: [SEP] is treated as the last subtoken
            #       of the last word for each segment
            segment_subtoken_map = (
                [segment_subtoken_map[0]]
                + segment_subtoken_map
                + [segment_subtoken_map[-1]]
            )
            # segment_mention_begin = [False] + segment_mention_begin + [False]
            # segment_mention_end = [False] + segment_mention_end + [False]
            segment_mention_begin = [0] + segment_mention_begin + [0]
            segment_mention_end = [0] + segment_mention_end + [0]

            segments.append(segment)
            segments_subtoken_map.append(segment_subtoken_map)
            segments_mention_begin.append(segment_mention_begin)
            segments_mention_end.append(segment_mention_end)

            curr_idx = end_idx + 1

        # Create a map from word index to subtoken indices (list)
        word_index_to_subtoken_indices \
            = self.get_word_index_to_subtoken_indices(
                segments_subtoken_map=segments_subtoken_map
            )

        # Subtoken index to word index
        subtoken_index_to_word_index = utils.flatten_lists(
            segments_subtoken_map
        )

        # Subtoken index to sentence index
        subtoken_index_to_sentence_index \
            = self.get_subtoken_index_to_sentence_index(
                segments=segments,
                doc_sentence_end=doc_sentence_end
            )

        # Subtoken indices for mention spans
        mention_begin_token_indices = utils.flatten_lists(
            [
                [subtok_i] * cnt
                for subtok_i, cnt in enumerate(
                    utils.flatten_lists(segments_mention_begin)
                )
            ]
        )
        mention_end_token_indices = utils.flatten_lists(
            [
                [subtok_i] * cnt
                for subtok_i, cnt in enumerate(
                    utils.flatten_lists(segments_mention_end)
                )
            ]
        )

        return (
            segments,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index,
            mention_begin_token_indices,
            mention_end_token_indices
        )

    def get_word_index_to_subtoken_indices(self, segments_subtoken_map):
        """
        Parameters
        ----------
        segments_subtoken_map : list[list[int]]

        Returns
        -------
        list[list[int]]
        """
        word_index_to_subtoken_indices = defaultdict(list)
        offset = 0
        for segment_subtoken_map in segments_subtoken_map:
            for subtok_i, word_i in enumerate(segment_subtoken_map):
                if subtok_i == 0 or subtok_i == len(segment_subtoken_map) - 1:
                    continue
                word_index_to_subtoken_indices[word_i].append(offset + subtok_i)
            offset += len(segment_subtoken_map)
        return word_index_to_subtoken_indices

    def get_subtoken_index_to_sentence_index(self, segments, doc_sentence_end):
        """
        Parameters
        ----------
        segments : list[list[str]]
        doc_sentence_end : list[bool]

        Returns
        -------
        list[int]
        """
        assert len(doc_sentence_end) == sum([len(seg) - 2 for seg in segments])
        sent_map = []
        sent_idx, subtok_idx = 0, 0
        for segment in segments:
            sent_map.append(sent_idx) # [CLS]
            length = len(segment) - 2
            for i in range(length):
                sent_map.append(sent_idx)
                sent_idx += int(doc_sentence_end[subtok_idx]) # 0 or 1
                subtok_idx += 1
            # [SEP] is the current sentence's last token
            sent_map.append(sent_idx - 1)
        return sent_map

    def convert_to_token_ids_with_padding(self, segments):
        """
        Parameters
        ----------
        segments: list[list[str]]

        Returns
        -------
        tuple[list[list[int]], list[list[int]]]
        """
        segments_id = [] # list[list[int]]
        segments_mask = [] # list[list[int]]
        n_subtokens = sum([len(s) for s in segments])
        for segment in segments:
            segment_id = self.tokenizer.convert_tokens_to_ids(segment)
            segment_mask = [1] * len(segment_id)
            while len(segment_id) < self.max_seg_len:
                segment_id.append(0)
                segment_mask.append(0)
            segments_id.append(segment_id)
            segments_mask.append(segment_mask)
        assert np.sum(np.asarray(segments_mask)) == n_subtokens
        return segments_id, segments_mask

