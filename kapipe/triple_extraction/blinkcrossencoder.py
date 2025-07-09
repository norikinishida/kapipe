from __future__ import annotations

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
import jsonlines

from . import shared_functions
from ..models import BlinkCrossEncoderModel
from .. import evaluation
from .. import utils
from ..utils import BestScoreHolder
from ..datatypes import (
    Config,
    Document,
    Mention,
    Entity,
    CandEntKeyInfo,
    CandidateEntitiesForDocument
)


logger = logging.getLogger(__name__)


class BlinkCrossEncoder:
    """
    BLINK Cross-Encoder (Wu et al., 2020).
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
        logger.info("########## BlinkCrossEncoder Initialization Starts ##########")

        self.device = device
        self.path_snapshot = path_snapshot

        if path_snapshot is not None:
            assert config is None
            assert path_entity_dict is None
            config = path_snapshot + "/config"
            path_entity_dict = path_snapshot + "/entity_dict.json"
            path_model = path_snapshot + "/model"

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
        if self.model_name == "blinkcrossencodermodel":
            self.model = BlinkCrossEncoderModel(
                device=device,
                bert_pretrained_name_or_path=config["bert_pretrained_name_or_path"],
                max_seg_len=config["max_seg_len"],
                entity_dict=self.entity_dict,
                mention_context_length=self.config["mention_context_length"]
            )
        else:
            raise Exception(f"Invalid model_name: {self.model_name}")

        # Show parameter shapes
        # logger.info("Model parameters:")
        # for name, param in self.model.named_parameters():
        #     logger.infof"{name}: {tuple(param.shape)}")

        # Load trained model parameters
        if path_snapshot is not None:
            self.model.load_state_dict(
                torch.load(path_model, map_location=torch.device("cpu")),
                strict=False
            )
            logger.info(f"Loaded model parameters from {path_model}")

        self.model.to(self.model.device)

        logger.info("########## BlinkCrossEncoder Initialization Ends ##########")

    def save(self, path_snapshot: str, model_only: bool = False) -> None:
        path_config = path_snapshot + "/config"
        path_entity_dict = path_snapshot + "/entity_dict.json"
        path_model = path_snapshot + "/model"
        if not model_only:
            utils.write_json(path_config, self.config)
            utils.write_json(path_entity_dict, list(self.entity_dict.values()))
        torch.save(self.model.state_dict(), path_model)

    def compute_loss(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument,
        mention_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert document["doc_key"] == candidate_entities_for_doc["doc_key"]

        # Switch to training mode
        self.model.train()

        # Preprocess
        preprocessed_data = self.model.preprocess(
            document=document,
            candidate_entities_for_doc=candidate_entities_for_doc,
            max_n_candidates=self.config["max_n_candidates_in_training"]
        )

        # Tensorize
        model_input = self.model.tensorize(
            preprocessed_data=preprocessed_data,
            mention_index=mention_index,
            compute_loss=True
        )

        # Forward
        model_output = self.model.forward(**model_input)

        return (
            model_output.loss,
            model_output.acc
        )

    def extract(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument
    ) -> Document:
        assert document["doc_key"] == candidate_entities_for_doc["doc_key"]

        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Return no entity if mention is missing
            if len(document["mentions"]) == 0:
                result_document = copy.deepcopy(document)
                result_document["entities"] = []
                return result_document

            # Preprocess
            preprocessed_data = self.model.preprocess(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc,
                max_n_candidates=self.config["max_n_candidates_in_inference"]
            )

            mentions: list[Mention] = []
            cands_for_mentions: list[list[CandEntKeyInfo]] = (
                candidate_entities_for_doc["candidate_entities"]
            )
            for mention_index in range(len(preprocessed_data["mentions"])):
                # Tensorize
                model_input = self.model.tensorize(
                    preprocessed_data=preprocessed_data,
                    mention_index=mention_index,
                    compute_loss=False
                )

                # Forward
                model_output = self.model.forward(**model_input)
                logits = model_output.logits # (1, n_candidates)

                # Structurize (1)
                # Transform logits to mention-level entity IDs
                pred_candidate_entity_index = torch.argmax(logits, dim=1).cpu().item() # int
                pred_candidate_entity_id = cands_for_mentions[mention_index][
                    pred_candidate_entity_index
                ]["entity_id"]
                mentions.append({"entity_id": pred_candidate_entity_id,})

            # Structurize (2)
            # Transform to entity-level entity IDs
            # i.e., aggregate mentions based on the entity IDs
            entities: list[Entity] = utils.aggregate_mentions_to_entities(
                document=document,
                mentions=mentions
            )

            # Integrate
            result_document = copy.deepcopy(document)
            for m_i in range(len(result_document["mentions"])):
                result_document["mentions"][m_i].update(mentions[m_i])
            result_document["entities"] = entities
            return result_document

    def batch_extract(
        self,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument]
    ) -> list[Document]:
        result_documents = []
        for document, candidate_entities_for_doc in tqdm(
            zip(documents, candidate_entities),
            total=len(documents),
            desc="extraction steps"
        ):
            result_document = self.extract(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc
            )
            result_documents.append(result_document)
        return result_documents


class BlinkCrossEncoderTrainer:
    """
    Trainer class for BLINK Cross-Encoder extractor.
    Handles training loop, evaluation, model saving, and early stopping.
    """

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str, str]:
        paths = {}
        # configurations
        paths["path_snapshot"] = self.base_output_path
        # training outputs
        paths["path_train_losses"] = self.base_output_path + "/train.losses.jsonl"
        paths["path_dev_evals"] = self.base_output_path + "/dev.eval.jsonl"
        # evaluation outputs
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"
        return paths

    def setup_dataset(
        self,
        extractor: BlinkCrossEncoder,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(extractor.entity_dict.keys()))
            gold_documents = []
            for document, candidate_entities_for_doc in tqdm(
                zip(documents, candidate_entities),
                desc="dataset setup"
            ):
                gold_doc = copy.deepcopy(document)

                cands_for_mentions: list[list[CandEntKeyInfo]] = (
                    candidate_entities_for_doc["candidate_entities"]
                )
                mentions: list[Mention] = document["mentions"]
                assert len(mentions) == len(cands_for_mentions)

                for m_i, (mention, cands_for_mention) in enumerate(zip(
                    mentions,
                    cands_for_mentions
                )):
                    cand_entity_ids = [c["entity_id"] for c in cands_for_mention]
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
        extractor: BlinkCrossEncoder,
        train_documents: list[Document],
        train_candidate_entities: list[CandidateEntitiesForDocument],
        dev_documents: list[Document],
        dev_candidate_entities: list[CandidateEntitiesForDocument]
    ) -> None:
        ##################
        # Setup
        ##################

        # Collect tuples of (document index, mention index, gold entity rank in candidates)
        train_doc_index_and_mention_index_tuples = [] # list[tuple[int,int,int]]
        train_doc_indices = np.arange(len(train_documents))
        for doc_i in train_doc_indices:
            ranks: list[int] = train_candidate_entities[doc_i][
                "original_gold_entity_rank_list"
            ]
            for m_i in range(len(train_documents[doc_i]["mentions"])):
                rank = ranks[m_i]
                train_doc_index_and_mention_index_tuples.append((doc_i, m_i, rank))

        # Sort the tuples based on their ranks in descending order
        train_doc_index_and_mention_index_tuples = sorted(
            train_doc_index_and_mention_index_tuples,
            key=lambda tpl: -tpl[-1]
        )
        train_doc_index_and_mention_index_tuples = np.asarray(
            train_doc_index_and_mention_index_tuples
        )

        # Limit the training instances to MAX_TRAINING_INSTANCES
        if extractor.config["max_training_instances"] is not None:
            n_prev_instances = len(train_doc_index_and_mention_index_tuples)
            train_doc_index_and_mention_index_tuples = (
                train_doc_index_and_mention_index_tuples[
                    :extractor.config["max_training_instances"]
                ]
            )
            n_new_instances = len(train_doc_index_and_mention_index_tuples)
            if n_prev_instances != n_new_instances:
                logger.info("Removed training mentions where the gold entity appears at a lower rank among the candidates")
                logger.info(f"{n_prev_instances} -> {n_new_instances} mentions")

        n_train = len(train_doc_index_and_mention_index_tuples)
        max_epoch = extractor.config["max_epoch"]
        batch_size = extractor.config["batch_size"]
        gradient_accumulation_steps = extractor.config["gradient_accumulation_steps"]
        total_update_steps = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * extractor.config["warmup_ratio"])

        logger.info("Number of training mentions: %d" % n_train)
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

        # Evaluate the extractor
        scores = self.evaluate(
            extractor=extractor,
            documents=dev_documents,
            candidate_entities=dev_candidate_entities,
            split="dev",
            #
            get_scores_only=True
        )
        scores.update({"epoch": 0, "step": 0})
        writer_dev.write(scores)
        logger.info(utils.pretty_format_dict(scores))

        # Set the best validation score
        bestscore_holder.compare_scores(
            scores["inkb_normalized_accuracy"]["accuracy"],
            0
        )

        # Save
        extractor.save(path_snapshot=self.paths["path_snapshot"])
        logger.info(f"Saved config, entity dictionary, and model to {self.paths['path_snapshot']}")

        ##################
        # Training Loop
        ##################

        bert_param, task_param = extractor.model.get_params()
        extractor.model.zero_grad()
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

                for (doc_i, mention_index, _) in (
                    train_doc_index_and_mention_index_tuples[
                        perm[instance_i : instance_i + batch_size]
                    ]
                ):
                    doc_i = int(doc_i)
                    mention_index = int(mention_index)

                    # Forward and compute loss
                    one_loss, one_acc = extractor.compute_loss(
                        document=train_documents[doc_i],
                        candidate_entities_for_doc=train_candidate_entities[doc_i],
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

                # Accumulate for reporting
                loss_accum += float(batch_loss.cpu())
                acc_accum += batch_acc
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
                        "accuracy": 100.0 * acc_accum / accum_count,
                        "max_valid_inkb_acc": bestscore_holder.best_score,
                        "patience": bestscore_holder.patience
                    }
                    writer_train.write(report)
                    logger.info(utils.pretty_format_dict(report))
                    loss_accum = 0.0
                    acc_accum = 0.0
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

                    # Evaluate the extractor
                    scores = self.evaluate(
                        extractor=extractor,
                        documents=dev_documents,
                        candidate_entities=dev_candidate_entities,
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
                    logger.info("[Step %d] Max validation InKB normalized accuracy: %f" % (step, bestscore_holder.best_score))

                    # Save the model
                    if did_update:
                        extractor.save(
                            path_snapshot=self.paths["path_snapshot"],
                            model_only=True
                        )
                        logger.info(f"Saved model to {self.paths['path_snapshot']}")

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
        extractor: BlinkCrossEncoder,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
        split: str,
        #
        prediction_only: bool = False,
        get_scores_only: bool = False,
    ) -> dict[str, Any] | None:
        # Apply the extractor
        result_documents = extractor.batch_extract(
            documents=documents,
            candidate_entities=candidate_entities
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)

        if prediction_only:
            return

        # Calculate the evaluation scores
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

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

