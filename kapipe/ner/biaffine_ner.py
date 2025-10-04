from __future__ import annotations

from collections import defaultdict
from collections import OrderedDict
import copy
import logging
import os
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
# from opt_einsum import contract
from tqdm import tqdm
import jsonlines

from ..datatypes import Config, Document, Mention
from .. import utils
from ..utils import BestScoreHolder
from .. import evaluation
from ..nn_utils import (
    Biaffine,
    FocalLoss,
    get_optimizer2,
    get_scheduler2
)


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
        if self.model_name == "biaffine_ner_model":
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

        optimizer = get_optimizer2(
            model=extractor.model,
            config=extractor.config
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


class MentionTuple(NamedTuple):
    span: tuple[int, int]
    name: str
    entity_type: str


class BiaffineNERModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        dropout_rate,
        vocab_etype,
        loss_function_name,
        focal_loss_gamma=None
    ):
        """
        Parameters
        ----------
        device : str
        bert_pretrained_name_or_path : str
        max_seg_len : int
        dropout_rate : float
        vocab_etype : dict[str, int]
        loss_function_name : str
        focal_loss_gamma : float | None
            by default None
        """
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.device = device
        self.bert_pretrained_name_or_path = bert_pretrained_name_or_path
        self.max_seg_len = max_seg_len
        self.dropout_rate = dropout_rate
        self.vocab_etype = vocab_etype
        self.loss_function_name = loss_function_name
        self.focal_loss_gamma = focal_loss_gamma

        self.n_entity_types = len(self.vocab_etype)

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert.config.hidden_size

        # Entity type classification
        self.linear_begin = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_end = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout_begin = nn.Dropout(p=self.dropout_rate)
        self.dropout_end = nn.Dropout(p=self.dropout_rate)
        self.biaffine = Biaffine(
            input_dim=self.hidden_dim,
            output_dim=self.n_entity_types,
            bias_x=True,
            bias_y=True
        )

        ######
        # Preprocessor
        ######

        self.preprocessor = BiaffineNERPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            vocab_etype=self.vocab_etype
        )

        ######
        # Loss Function
        ######

        if self.loss_function_name == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(reduction="none")
        elif self.loss_function_name == "focal_loss":
            self.loss_function = FocalLoss(
                gamma=self.focal_loss_gamma,
                reduction="none"
            )
        else:
            raise Exception(
                f"Invalid loss_function: {self.loss_function_name}"
            )

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
    # Forward pass
    ################

    def preprocess(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess(document=document)

    def tensorize(self, preprocessed_data, compute_loss):
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

        if not compute_loss:
            return model_input

        # (n_tokens, n_tokens)
        model_input["matrix_valid_span_mask"] = torch.tensor(
            preprocessed_data["matrix_valid_span_mask"],
            device=self.device
        ).to(torch.float)

        # (n_tokens, n_tokens)
        model_input["matrix_gold_entity_type_labels"] = torch.tensor(
            preprocessed_data["matrix_gold_entity_type_labels"],
            device=self.device
        ).to(torch.long)

        return model_input

    def forward(
        self,
        segments_id,
        segments_mask,
        compute_loss,
        matrix_valid_span_mask=None,
        matrix_gold_entity_type_labels=None
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        compute_loss : bool
        matrix_valid_span_mask : torch.Tensor | None
            shape of (n_tokens, n_tokens); by default None
        matrix_gold_entity_type_labels : torch.Tensor | None
            shape of (n_tokens, n_tokens); by default None

        Returns
        -------
        ModelOutput
        """
        # Encode tokens by BERT
        # (n_tokens, hidden_dim)
        token_vectors = self.encode_tokens(
            segments_id=segments_id,
            segments_mask=segments_mask
        )

        # Compute logits by Biaffine
        # (n_tokens, n_tokens, n_entity_types)
        logits = self.compute_logits_by_biaffine(
            token_vectors=token_vectors
        )

        if not compute_loss:
            return ModelOutput(
                logits=logits
            )

        # Compute loss (summed over valid spans)
        # (n_tokens, n_tokens)
        loss = self.loss_function(
            logits.permute(2,0,1).unsqueeze(0),
            matrix_gold_entity_type_labels.unsqueeze(0)
        ).squeeze(0)
        loss = loss * matrix_valid_span_mask
        loss = loss.sum() # Scalar

        # Compute accuracy (summed over valid spans)
        # (n_tokens, n_tokens)
        matrix_pred_entity_type_labels = logits.argmax(dim=-1)
        # (n_tokens, n_tokens)
        acc = (
            matrix_pred_entity_type_labels == matrix_gold_entity_type_labels
        ).to(torch.float)
        acc = acc * matrix_valid_span_mask
        acc = acc.sum().item() # Scalar

        n_valid_spans = int(matrix_valid_span_mask.sum().item())

        return ModelOutput(
            logits=logits,
            loss=loss,
            acc=acc,
            n_valid_spans=n_valid_spans
        )

    ################
    # Subfunctions
    ################

    def encode_tokens(self, segments_id, segments_mask):
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
        # Check
        n_segments, max_seg_len = segments_id.shape
        assert max_seg_len == self.max_seg_len

        # Encode segments by BERT
        bert_output = self.bert(
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

    def compute_logits_by_biaffine(self, token_vectors):
        """
        Parameters
        ----------
        token_vectors : torch.Tensor
            shape of (n_tokens, hidden_dim)

        Returns
        -------
        torch.Tensor
            shape of (n_tokens, n_tokens, n_entity_types)
        """
        # (n_tokens, hidden_dim)
        zb = self.dropout_begin(
            torch.tanh(
                self.linear_begin(token_vectors)
            )
        )
        # (n_tokens, hidden_dim)
        ze = self.dropout_end(
            torch.tanh(
                self.linear_end(token_vectors)
            )
        )
        # (batch_size=1, n_tokens, hidden_dim)
        zb = zb.unsqueeze(0)
        # (batch_size=1, n_tokens, hidden_dim)
        ze = ze.unsqueeze(0)
        # (batch_size=1, n_entity_types, n_tokens, n_tokens)
        logits = self.biaffine(zb.float(), ze.float())
        # (batch_size=1, n_tokens, n_tokens, n_entity_types)
        logits = logits.permute(0, 2, 3, 1)
        # (n_tokens, n_tokens, n_entity_types)
        logits = logits.squeeze(0)
        return logits


class BiaffineNERPreprocessor:

    def __init__(self, tokenizer, max_seg_len, vocab_etype):
        """
        Parameters
        ----------
        tokenizer: PreTrainedTokenizer
        max_seg_len: int
        vocab_etype: dict[str, int]
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.vocab_etype = vocab_etype

        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token

    # ---

    def preprocess(self, document):
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
        #####

        preprocessed_data["doc_key"] = document["doc_key"]

        sentences = [s.split() for s in document["sentences"]]
        preprocessed_data["sentences"] = sentences

        with_supervision = True if "mentions" in document else False
        if with_supervision:
            mentions = [
                MentionTuple(
                    tuple(m["span"]),
                    m["name"],
                    m["entity_type"]
                )
                for m in document["mentions"]
            ]
            preprocessed_data["mentions"] = mentions

        #####
        # bert_input: dict[str, Any]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        # subtoken_index_to_word_index: list[int]
        # word_index_to_subtoken_indices: list[ist[int]]
        # subtoken_index_to_sentence_index: list[int]
        #####

        (
            segments,
            segments_id,
            segments_mask,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index
        ) = self.tokenize_and_split(sentences=sentences)
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

        #####
        # matrix_valid_span_mask: list[list[float]]
        #####

        # We will get prediction and losses only for spans
        #   within the same sentence.
        n_subtokens = len(utils.flatten_lists(bert_input["segments"]))
        matrix_valid_span_mask = np.zeros((n_subtokens, n_subtokens))
        offset = 0
        for sent in sentences:
            for local_word_i in range(0, len(sent)):
                global_word_i = offset + local_word_i
                # first subtoken
                global_subtok_i \
                    = word_index_to_subtoken_indices[global_word_i][0]
                for local_word_j in range(local_word_i, len(sent)):
                    global_word_j = offset + local_word_j
                    # first subtoken
                    global_subtok_j \
                        = word_index_to_subtoken_indices[global_word_j][0]
                    matrix_valid_span_mask[global_subtok_i, global_subtok_j] \
                        = 1.0
            offset += len(sent)
        preprocessed_data["matrix_valid_span_mask"] = matrix_valid_span_mask

        #####
        # matrix_gold_entity_type_labels: list[list[int]]
        #####

        if with_supervision:
            matrix_gold_entity_type_labels = np.zeros(
                (n_subtokens, n_subtokens), dtype=np.int32
            )
            for mention in mentions:
                begin_word_i, end_word_i = mention.span
                begin_subtok_i = word_index_to_subtoken_indices[begin_word_i][0]
                end_subtok_i = word_index_to_subtoken_indices[end_word_i][0]
                etype_id = self.vocab_etype[mention.entity_type] # str -> int
                matrix_gold_entity_type_labels[begin_subtok_i, end_subtok_i] \
                    = etype_id
            preprocessed_data["matrix_gold_entity_type_labels"] \
                = matrix_gold_entity_type_labels

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(self, sentences):
        """
        Parameters
        ----------
        sentences: list[list[str]]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]], list[int],
            list[list[int]], list[int]]
        """
        # subtoken分割、subtoken単位でのtoken終点、文終点のindicatorsの作成
        (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index
        ) = self.tokenize(sentences=sentences)

        # 1階層のリストに変換
        doc_subtoken = utils.flatten_lists(sents_subtoken) # list[str]
        doc_token_end = utils.flatten_lists(sents_token_end) # list[bool]
        doc_sentence_end = utils.flatten_lists(sents_sentence_end) # list[bool]

        # BERTセグメントに分割
        (
            segments,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index
        ) = self.split(
             doc_subtoken=doc_subtoken,
             doc_sentence_end=doc_sentence_end,
             doc_token_end=doc_token_end,
             subtoken_index_to_word_index=subtoken_index_to_word_index
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
            subtoken_index_to_sentence_index
        )

    def tokenize(self, sentences):
        """
        Parameters
        ----------
        sentences: list[list[str]]

        Returns
        -------
        tuple[list[list[str]], list[list[bool]], list[list[bool]], list[int]]
        """
        sents_subtoken = [] # list[list[str]]
        sents_token_end = [] # list[list[bool]]
        sents_sentence_end = [] # list[list[bool]]
        subtoken_index_to_word_index = [] # list[int]

        word_idx = -1
        offset = 0
        for sent in sentences:
            sent_subtoken = [] # list[str]
            sent_token_end = [] # list[bool]
            sent_sentence_end = [] # list[bool]
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
            # 文の終了位置
            sent_sentence_end[-1] = True
            sents_subtoken.append(sent_subtoken)
            sents_token_end.append(sent_token_end)
            sents_sentence_end.append(sent_sentence_end)
            offset += len(sent)

        return (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index
        )

    def split(
        self,
        doc_subtoken,
        doc_sentence_end,
        doc_token_end,
        subtoken_index_to_word_index
    ):
        """
        Parameters
        ----------
        doc_subtoken: list[str]
        doc_sentence_end: list[bool]
        doc_token_end: list[bool]
        subtoken_index_to_word_index: list[int]

        Returns
        -------
        tuple[list[list[str]], list[int], list[list[int]], list[int]]
        """
        segments = [] # list[list[str]]
        segments_subtoken_map = [] # list[list[int]]

        n_subtokens = len(doc_subtoken)
        curr_idx = 0 # Index for subtokens
        while curr_idx < len(doc_subtoken):
            # Try to split at a sentence end point
            end_idx = min(curr_idx + self.max_seg_len - 1 - 2, n_subtokens - 1)
            while end_idx >= curr_idx and not doc_sentence_end[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.warning("No sentence end found; split at token end")
                # If no sentence end point found,
                #   try to split at token end point.
                end_idx = min(
                    curr_idx + self.max_seg_len - 1 - 2,
                    n_subtokens - 1
                )
                seg_before \
                    = "\"" + " ".join(doc_subtoken[curr_idx:end_idx+1]) + "\""
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
            segment_subtoken_map \
                = subtoken_index_to_word_index[curr_idx: end_idx + 1]
            segment = [self.cls_token] + segment + [self.sep_token]
            # NOTE: [CLS] is treated as the first subtoken
            #     of the first word for each segment.
            # NOTE: [SEP] is treated as the last subtoken
            #     of the last word for each segment.
            segment_subtoken_map = (
                [segment_subtoken_map[0]]
                + segment_subtoken_map
                + [segment_subtoken_map[-1]]
            )

            segments.append(segment)
            segments_subtoken_map.append(segment_subtoken_map)

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

        return (
            segments,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index
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
        sent_map = [] # list[int]
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

    def convert_to_token_ids_with_padding(
        self,
        segments
    ):
        """
        Parameters
        ----------
        segments: list[list[str]]

        Returns
        -------
        Tuple[list[list[int]], list[list[int]]]
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


