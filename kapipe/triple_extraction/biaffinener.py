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
from ..models import BiaffineNERModel
from .. import evaluation
from .. import utils
from ..utils import BestScoreHolder
from ..datatypes import Config, Document, Mention


logger = logging.getLogger(__name__)


class BiaffineNER:
    """
    Biaffine Named Entity Recognizer (Yu et al., 2020).
    """

    def __init__(
        self,
        device: str,
        # Initialization
        config: Config | str | None = None,
        vocab_etype: dict[str, int] | str | None = None,
        # Loading
        path_snapshot: str | None = None
    ):
        logger.info("########## BiaffineNER Initialization Starts ##########")

        self.device = device
        self.path_snapshot = path_snapshot

        if path_snapshot is not None:
            assert config is None
            assert vocab_etype is None
            config = path_snapshot + "/config"
            vocab_etype = path_snapshot + "/entity_types.vocab.txt"
            path_model = path_snapshot + "/model"

        # Load the configuration
        if isinstance(config, str):
            config_path = config
            config = utils.get_hocon_config(config_path=config_path)
            logger.info(f"Loaded configuration from {config_path}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

        # Load the entity type vocabulary
        if isinstance(vocab_etype, str):
            vocab_path = vocab_etype
            vocab_etype = utils.read_vocab(vocab_path)
            logger.info(f"Loaded entity type vocabulary from {vocab_path}")
        self.vocab_etype = vocab_etype
        self.ivocab_etype = {i: l for l, i in self.vocab_etype.items()}

        # Initialize the model
        self.model_name = self.config["model_name"]
        if self.model_name == "biaffinenermodel":
            self.model = BiaffineNERModel(
                device=device,
                bert_pretrained_name_or_path=config["bert_pretrained_name_or_path"],
                max_seg_len=config["max_seg_len"],
                dropout_rate=config["dropout_rate"],
                vocab_etype=self.vocab_etype,
                loss_function_name=config["loss_function"],
                focal_loss_gamma=(
                    config["focal_loss_gamma"]
                    if config["loss_function"] == "focal_loss" else None
                )
            )
        else:
            raise ValueError(f"Invalid model_name: {self.model_name}")

        # Show parameter shapes
        # logger.info("Model parameters:")
        # for name, param in self.model.named_parameters():
        #     logger.info(f"{name}: {tuple(param.shape)}")

        # Load trained model parameters
        if path_snapshot is not None:
            self.model.load_state_dict(
                torch.load(path_model, map_location=torch.device("cpu")),
                strict=False
            )
            logger.info(f"Loaded model parameters from {path_model}")

        self.model.to(self.model.device)

        # Initialize the span-based decoder
        self.decoder = SpanBasedDecoder(
            allow_nested_entities=self.config["allow_nested_entities"]
        )

        logger.info("########## BiaffineNER Initialization Ends ##########")

    def save(self, path_snapshot: str, model_only: bool = False) -> None:
        path_config = path_snapshot + "/config"
        path_vocab = path_snapshot + "/entity_types.vocab.txt"
        path_model = path_snapshot + "/model"
        if not model_only:
            utils.write_json(path_config, self.config)
            utils.write_vocab(path_vocab, self.vocab_etype, write_frequency=False)
        torch.save(self.model.state_dict(), path_model)

    def compute_loss(self, document: Document) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Switch to training mode
        self.model.train()

        # Preprocess
        preprocessed_data = self.model.preprocess(document=document)

        # Tensorize
        model_input = self.model.tensorize(
            preprocessed_data=preprocessed_data,
            compute_loss=True
        )

        # Forward
        model_output = self.model.forward(**model_input)

        return (
            model_output.loss,
            model_output.acc,
            model_output.n_valid_spans
        )

    def extract(self, document: Document) -> Document:
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Preprocess
            preprocessed_data = self.model.preprocess(document=document)

            # Tensorize
            model_input = self.model.tensorize(
                preprocessed_data=preprocessed_data,
                compute_loss=False
            )

            # Forward
            model_output = self.model.forward(**model_input)
            logits = model_output.logits # (n_tokens, n_tokens, n_etypes)

            # Structurize
            mentions = self.structurize(
                document=document,
                logits=logits,
                matrix_valid_span_mask=preprocessed_data["matrix_valid_span_mask"],
                subtoken_index_to_word_index=preprocessed_data["bert_input"]["subtoken_index_to_word_index"]
            )

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["mentions"] = mentions
            return result_document

    def structurize(
        self,
        document: Document,
        logits: torch.Tensor,
        matrix_valid_span_mask: np.ndarray,
        subtoken_index_to_word_index: list[int]
    ) -> list[Mention]:
        # Transform logits to prediction scores and labels for each token-token pair
        # (n_tokens, n_tokens), (n_tokens, n_tokens)
        matrix_pred_entity_type_scores, matrix_pred_entity_type_labels = logits.max(dim=-1)
        matrix_pred_entity_type_scores= matrix_pred_entity_type_scores.cpu().numpy()
        matrix_pred_entity_type_labels = matrix_pred_entity_type_labels.cpu().numpy()

        # Apply mask to invalid token-token pairs
        # NOTE: The "NON-ENTITY" class corresponds to the 0th label
        # (n_tokens, n_tokens)
        matrix_pred_entity_type_labels = matrix_pred_entity_type_labels * matrix_valid_span_mask

        # Get spans that have non-zero entity type label
        # (n_spans,), (n_spans,)
        span_begin_token_indices, span_end_token_indices = np.nonzero(
            matrix_pred_entity_type_labels
        )
        # (n_spans,)
        span_entity_type_scores = matrix_pred_entity_type_scores[
            span_begin_token_indices, span_end_token_indices
        ].tolist()
        # (n_spans,)
        span_entity_type_labels = matrix_pred_entity_type_labels[
            span_begin_token_indices, span_end_token_indices
        ].tolist()
        # (n_spans,)
        span_entity_types = [self.ivocab_etype[etype_i] for etype_i in span_entity_type_labels]

        # Transform the subtoken-level spans to word-level spans
        # (n_spans,)
        span_begin_token_indices = [
            subtoken_index_to_word_index[subtok_i]
            for subtok_i in span_begin_token_indices
        ]
        # (n_spans,)
        span_end_token_indices = [
            subtoken_index_to_word_index[subtok_i]
            for subtok_i in span_end_token_indices
        ]

        # Apply filtering
        spans = list(zip(
            span_begin_token_indices,
            span_end_token_indices,
            span_entity_types,
            span_entity_type_scores
        ))
        # Remove too-long spans (possibly predicted spans)
        spans = [(b,e,t,s) for b,e,t,s in spans if (e - b) <= 10]

        # Decode into mention format
        words = " ".join(document["sentences"]).split()
        mentions = self.decoder.decode(spans=spans, words=words)

        return mentions

    def batch_extract(self, documents: list[Document]) -> list[Document]:
        result_documents = []
        for document in tqdm(documents, desc="extraction steps"):
            result_document = self.extract(document=document)
            result_documents.append(result_document)
        return result_documents


class SpanBasedDecoder:
    """
    A span-based decoder for Named Entity Recognition.

    It selects valid spans from model predictions and decodes them into mention objects.
    Supports both Flat and Nested NER based on configuration.
    """

    def __init__(self, allow_nested_entities: bool):
        self.allow_nested_entities = allow_nested_entities

    def decode(
        self,
        spans: list[tuple[int, int, str, float]],
        words: list[str]
    ) -> list[Mention]:
        mentions: list[Mention] = []

        # Sort the candidate spans by scores (descending)
        spans = sorted(spans, key=lambda x: -x[-1])

        # Select spans
        n_words = len(words)
        self.check_matrix = np.zeros((n_words, n_words)) # Used in Flat NER
        self.check_set = set() # Used in Nested NER
        for span in spans:
            begin_token_index, end_token_index, etype, _ = span
            name = " ".join(words[begin_token_index: end_token_index + 1])
            if self.is_violation(
                begin_token_index=begin_token_index,
                end_token_index=end_token_index
            ):
                continue
            mentions.append({
                "span": (begin_token_index, end_token_index),
                "name": name,
                "entity_type": etype,
            })
            self.check_matrix[begin_token_index: end_token_index + 1] = 1
            self.check_set.add((begin_token_index, end_token_index))

        # Sort mentions by span position
        mentions = sorted(mentions, key=lambda m: m["span"])

        return mentions

    def is_violation(self, begin_token_index: int, end_token_index: int) -> bool:
        if not self.allow_nested_entities:
            # Flat NER
            if self.check_matrix[begin_token_index: end_token_index + 1].sum() > 0:
                return True
            return False
        else:
            # Nested NER
            for begin_token_j, end_token_j in self.check_set:
                if (
                    (begin_token_index < begin_token_j <= end_token_index < end_token_j)
                    or
                    (begin_token_j < begin_token_index <= end_token_j < end_token_index)
                ):
                    return True
            return False


class BiaffineNERTrainer:
    """
    Trainer class for BiaffineNER extractor.
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
            "path_dev_eval": f"{self.base_output_path}/dev.eval.json",
            "path_test_gold": f"{self.base_output_path}/test.gold.json",
            "path_test_pred": f"{self.base_output_path}/test.pred.json",
            "path_test_eval": f"{self.base_output_path}/test.eval.json"
        }

    def setup_dataset(
        self,
        extractor: BiaffineNER,
        documents: list[Document],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            gold_documents = [
                copy.deepcopy(doc)
                for doc in tqdm(documents, desc="dataset setup")
            ]
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def train(
        self,
        extractor: BiaffineNER,
        train_documents: list[Document],
        dev_documents: list[Document]
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
        bestscore_holder.compare_scores(scores["span_and_type"]["f1"], 0)

        # Save
        extractor.save(path_snapshot=self.paths["path_snapshot"])
        logger.info(f"Saved config, vocab, and model to {self.paths['path_snapshot']}")

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
                actual_total_spans = 0

                for doc_i in train_doc_indices[perm[instance_i: instance_i + batch_size]]:
                    # Forward and compute loss
                    (
                        one_loss,
                        one_acc,
                        n_valid_spans
                    ) = extractor.compute_loss(
                        document=train_documents[doc_i]
                    )

                    # Accumulate the loss
                    batch_loss = batch_loss + one_loss
                    batch_acc += one_acc
                    actual_batchsize += 1
                    actual_total_spans += n_valid_spans

                # Average the loss
                actual_batchsize = float(actual_batchsize)
                actual_total_spans = float(actual_total_spans)
                batch_loss = batch_loss / actual_total_spans # loss per span
                batch_acc = batch_acc / actual_total_spans

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
                        "step_progress": f"{step}/{total_update_steps}",
                        "step_progress(ratio)": 100.0 * step / total_update_steps,
                        "one_epoch_progress": f"{instance_i + batch_size}/{n_train}",
                        "one_epoch_progress(ratio)": 100.0 * (instance_i + batch_size) / n_train,
                        "loss": loss_accum / accum_count,
                        "accuracy": 100.0 * acc_accum / accum_count,
                        "max_valid_f1": bestscore_holder.best_score,
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
                        split="dev",
                        #
                        get_scores_only=True
                    )
                    scores.update({"epoch": epoch, "step": step})
                    writer_dev.write(scores)
                    logger.info(utils.pretty_format_dict(scores))

                    # Update the best validation score
                    did_update = bestscore_holder.compare_scores(
                        scores["span_and_type"]["f1"],
                        epoch
                    )

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
        extractor: BiaffineNER,
        documents: list[Document],
        split: str,
        #
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:
        # Apply the extractor
        result_documents = extractor.batch_extract(documents=documents)
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)

        if prediction_only:
            return

        # Calculate the evaluation scores
        scores = evaluation.ner.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"]
        )
        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
