from __future__ import annotations

import copy
import logging
import os
import random
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
import jsonlines

from . import shared_functions
from ..models import BlinkBiEncoderModel
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
        if self.model_name == "blinkbiencodermodel":
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
        # The GPU ID for indexing should NOT be the same with the GPU ID of the BLINK model to avoid OOM error
        # Here, we assume that GPU-0 is set for the BLINK model.
        self.anns = ApproximateNearestNeighborSearch(gpu_id=1) # TODO: Allow GPU-ID selection

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
                "id": entity_id,
                "title": canonical_name,
                "text": description,
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
                    "id": entity_id,
                    "title": canonical_name,
                    "text": description,
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
                passage_ids=[p["id"] for p in entity_passages],
                passage_metadatas=[{"title": p["title"]} for p in entity_passages]
            )

            self.precomputed_entity_vectors = entity_vectors

            end_time = time.time()
            span_time = end_time - start_time
            span_time /= 60.0
            logger.info("Completed indexing")
            logger.info(f"Time: {span_time} min.")

    def extract(self, document: Document, retrieval_size: int = 1) -> tuple[
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
            #   (n_mentions, retrieval_size),
            #   (n_mentions, retrieval_size)
            (
                _,
                mention_pred_entity_ids,
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

    def batch_extract(
        self,
        documents: list[Document],
        retrieval_size: int = 1
    ) -> tuple[list[Document], list[CandidateEntitiesForDocument]]:
        result_documents: list[Document] = []
        candidate_entities: list[CandidateEntitiesForDocument] = []
        for document in tqdm(documents, desc="extraction steps"):
            result_document, candidate_entities_for_doc = self.extract(
                document=document,
                retrieval_size=retrieval_size
            )
            result_documents.append(result_document)
            candidate_entities.append(candidate_entities_for_doc)
        return result_documents, candidate_entities


class BlinkBiEncoderTrainer:
    """
    Trainer class for Blink Bi-Encoder extractor.
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
        extractor: BlinkBiEncoder,
        documents: list[Document],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            # Extract all concepts from the entity dictionary
            kb_entity_ids = set(list(extractor.entity_dict.keys()))
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
        extractor: BlinkBiEncoder,
        train_documents: list[Document],
        dev_documents: list[Document],
    ) -> None:
        ##################
        # Setup
        ##################

        train_doc_indices = np.arange(len(train_documents))

        n_train = len(train_doc_indices)
        max_epoch = extractor.config["max_epoch"]
        batch_size = extractor.config["batch_size"]
        gradient_accumulation_steps = extractor.config["gradient_accumulation_steps"]
        total_update_steps = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
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
        scheduler = shared_functions.get_scheduler2(
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
        extractor.make_index()

        # Evaluate the extractor
        scores = self.evaluate(
            extractor=extractor,
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
        extractor.save(path_snapshot=self.paths["path_snapshot"])
        logger.info(f"Saved config, entity dictionary, model, and entity vectors to {self.paths['path_snapshot']}")

        ##################
        # Training Loop
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

                for doc_i in train_doc_indices[perm[instance_i: instance_i + batch_size]]:
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

                    if extractor.config["max_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            bert_param,
                            extractor.config["max_grad_norm"]
                        )
                        torch.nn.utils.clip_grad_norm_(
                            task_param,
                            extractor.config["max_grad_norm"]
                        )

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
                        (extractor.config["n_steps_for_validation"] > 0)
                        and
                        (step % extractor.config["n_steps_for_validation"] == 0)
                    )
                ):

                    ##################
                    # Validation
                    ##################

                    # Build index
                    extractor.make_index()

                    # Evaluate the extractor
                    scores = self.evaluate(
                        extractor=extractor,
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
                        extractor.save(
                            path_snapshot=self.paths["path_snapshot"],
                            model_only=True
                        )
                        logger.info(f"Saved model and entity vectors to {self.paths['path_snapshot']}")

                    ##################
                    # Termination Check
                    ##################

                    if bestscore_holder.patience >= extractor.config["max_patience"]:
                        writer_train.close()
                        writer_dev.close()
                        progress_bar.close()
                        return

        writer_train.close()
        writer_dev.close()
        progress_bar.close()

    def evaluate(
        self,
        extractor: BlinkBiEncoder,
        documents: list[Document],
        split: str,
        #
        prediction_only: bool = False,
        get_scores_only: bool = False,
    ) -> dict[str, Any] | None:
        # Apply the extractor
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
        extractor: BlinkBiEncoder,
        documents: list[Document]
    ) -> list[dict[str, list[CandEntKeyInfo]]]:
        logger.info("Generating candidate entities for training ...")
        start_time = time.time()

        RETRIEVAL_SIZE = 10 # the number of retrieved entities for each mention

        flatten_candidate_entities: list[dict[str, list[CandEntKeyInfo]]] = []

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
            # Aggregate gold entities for the mentions in the document
            gold_entity_ids = list(set([m["entity_id"] for m in document["mentions"]]))
            assert len(gold_entity_ids) <= extractor.config["n_candidate_entities"]

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
            tuples = tuples[:extractor.config["n_candidate_entities"]]

            # Sample entities randomly if the number of candidates is less than the specified number
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

