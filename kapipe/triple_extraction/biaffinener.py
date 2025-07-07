import copy
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
import jsonlines

from . import shared_functions
from ..models import BiaffineNERModel
from .. import evaluation
from .. import utils
from ..utils import BestScoreHolder


logger = logging.getLogger(__name__)


class BiaffineNER:
    """ Biaffine-NER (Yu et al., 2020).
    """

    def __init__(
        self,
        # General
        device,
        config,
        # Task specific
        vocab_etype,
        # Misc.
        path_model=None,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        vocab_etype: dict[str, int] | str
        path_model: str | None
            by default None
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> BiaffineNER Initialization >>>>>>>>>>")
        self.device = device

        ######
        # Config
        ######

        if isinstance(config, str):
            tmp = config
            config = utils.get_hocon_config(
                config_path=config,
                config_name=None
            )
            if self.verbose:
                logger.info(f"Loaded configuration from {tmp}")
        self.config = config
        if self.verbose:
            logger.info(utils.pretty_format_dict(self.config))

        ######
        # Vocabulary (entity types)
        ######

        if isinstance(vocab_etype, str):
            tmp = vocab_etype
            vocab_etype = utils.read_vocab(vocab_etype)
            if self.verbose:
                logger.info(f"Loaded entity type vocabulary from {tmp}")
        self.vocab_etype = vocab_etype
        self.ivocab_etype = {i:l for l, i in self.vocab_etype.items()}

        ######
        # Model
        ######

        self.model_name = self.config["model_name"]
        if self.model_name == "biaffinenermodel":
            self.model = BiaffineNERModel(
                device=device,
                bert_pretrained_name_or_path=config[
                    "bert_pretrained_name_or_path"
                ],
                max_seg_len=config["max_seg_len"],
                dropout_rate=config["dropout_rate"],
                vocab_etype=self.vocab_etype,
                loss_function_name=config["loss_function"],
                focal_loss_gamma=config["focal_loss_gamma"] \
                    if config["loss_function"] == "focal_loss" else None
            )
        else:
            raise Exception(f"Invalid model_name: {self.model_name}")

        # Show parameter shapes
        # logger.info("Model parameters:")
        # for name, param in self.model.named_parameters():
        #     logger.info(f"{name}: {tuple(param.shape)}")

        # Load trained model parameters
        if path_model is not None:
            self.load_model(path=path_model)
            if self.verbose:
                logger.info(f"Loaded model parameters from {path_model}")

        self.model.to(self.model.device)

        ######
        # Decoder
        ######

        self.decoder = SpanBasedDecoder(
            allow_nested_entities=self.config["allow_nested_entities"]
        )

        if self.verbose:
            logger.info("<<<<<<<<<< BiaffineNER Initialization <<<<<<<<<<")

    def load_model(self, path):
        """
        Parameters
        ----------
        path : str
        """
        self.model.load_state_dict(
            torch.load(path, map_location=torch.device("cpu")),
            strict=False
        )

    def save_model(self, path):
        """
        Parameters
        ----------
        path : str
        """
        torch.save(self.model.state_dict(), path)

    # ---

    def compute_loss(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, int]
        """
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

    def extract(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        Document
        """
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
            # (n_tokens, n_tokens, n_etypes)
            logits = model_output.logits

            # Structurize
            mentions = self.structurize(
                document=document,
                logits=logits,
                matrix_valid_span_mask=\
                    preprocessed_data["matrix_valid_span_mask"],
                subtoken_index_to_word_index=\
                    preprocessed_data["bert_input"]["subtoken_index_to_word_index"]
            )

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["mentions"] = mentions
            return result_document

    def structurize(
        self,
        document,
        logits,
        matrix_valid_span_mask,
        subtoken_index_to_word_index
    ):
        # Transform logits to prediction scores and labels
        #   for each token-token pair.
        # (n_tokens, n_tokens), (n_tokens, n_tokens)
        matrix_pred_entity_type_scores, matrix_pred_entity_type_labels \
            = logits.max(dim=-1)
        matrix_pred_entity_type_scores \
            = matrix_pred_entity_type_scores.cpu().numpy()
        matrix_pred_entity_type_labels \
            = matrix_pred_entity_type_labels.cpu().numpy()

        # Apply mask to invalid token-token pairs
        # NOTE: The "NON-ENTITY" class corresponds to the 0th label
        # (n_tokens, n_tokens)
        matrix_pred_entity_type_labels = (
            matrix_pred_entity_type_labels * matrix_valid_span_mask
        )

        # Get spans that have non-zero entity type label
        # (n_spans,), (n_spans,)
        span_begin_token_indices, span_end_token_indices \
            = np.nonzero(matrix_pred_entity_type_labels)
        # (n_spans,)
        span_entity_type_scores = matrix_pred_entity_type_scores[
            span_begin_token_indices, span_end_token_indices
        ].tolist()
        # (n_spans,)
        span_entity_type_labels = matrix_pred_entity_type_labels[
            span_begin_token_indices, span_end_token_indices
        ].tolist()
        # (n_spans,)
        span_entity_types = [
            self.ivocab_etype[etype_i]
            for etype_i in span_entity_type_labels
        ]

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
        mentions = self.decoder.decode(
            spans=spans,
            # words=utils.flatten_lists(preprocessed_data["sentences"])
            words=" ".join(document["sentences"]).split()
        )

        return mentions

    def batch_extract(self, documents):
        """
        Parameters
        ----------
        documents : list[Document]

        Returns
        -------
        list[Document]
        """
        result_documents = []
        for document in tqdm(documents, desc="extraction steps"):
            result_document = self.extract(
                document=document
            )
            result_documents.append(result_document)
        return result_documents


class SpanBasedDecoder:
    """A span-based decoder for NER. This decoder applies constraints (for either Flat NER or Nested NER) to given scored spans and outputs a list of mentions.
    """

    def __init__(
        self,
        allow_nested_entities
    ):
        """
        Parameters
        ----------
        allow_nested_entities : bool
        """
        self.allow_nested_entities = allow_nested_entities

    def decode(
        self,
        spans,
        words
    ):
        """
        Parameters
        ----------
        spans : list[tuple[int, int, str, float]]
        words : list[str]

        Results
        -------
        list[Mention]
        """
        mentions = [] # list[Mention]

        # Sort the candidate spans based on their scores
        spans = sorted(spans, key=lambda x: -x[-1])

        # Select spans
        n_words = len(words)
        self.check_matrix = np.zeros((n_words, n_words)) # Used in Flat NER
        self.check_set = set() # Uased in Nested NER
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

        # Sort mentions based on the positions
        mentions = sorted(mentions, key=lambda m: m["span"])

        return mentions

    def is_violation(self, begin_token_index, end_token_index):
        """
        Parameters
        ----------
        begin_token_index : int
        end_token_index : int

        Returns
        -------
        bool
        """
        if not self.allow_nested_entities:
            # Flat NER
            if (
                self.check_matrix[begin_token_index: end_token_index + 1].sum()
                > 0
            ):
                return True
            else:
                return False
        else:
            # Nested NER
            for begin_token_j, end_token_j in self.check_set:
                if (
                    (begin_token_index < begin_token_j
                     <= end_token_index < end_token_j)
                    or
                    (begin_token_j < begin_token_index
                     <= end_token_j < end_token_index)
                ):
                    return True
            return False


class BiaffineNERTrainer:

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
        # Path to vocabulary
        paths["path_vocab_etype"] = self.base_output_path + "/entity_types.vocab.txt"
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

    def setup_dataset(self, extractor, documents, split):
        """
        Parameters
        ----------
        extractor : BiaffineNER
        documents : list[Document]
        split : str
        """
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            gold_documents = []
            for document in tqdm(documents, desc="dataset setup"):
                gold_doc = copy.deepcopy(document)
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def train(self, extractor, train_documents, dev_documents):
        """
        Parameters
        ----------
        extractor : BiaffineNER
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

        bestscore_holder.compare_scores(scores["span_and_type"]["f1"], 0)

        ##################
        # Save
        ##################

        # Save the model
        extractor.save_model(path=self.paths["path_snapshot"])
        logger.info("Saved model to %s" % self.paths["path_snapshot"])

        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], extractor.config)
        utils.write_json(self.paths["path_config"], extractor.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

        # Save the vocabulary (only once)
        utils.write_vocab(
            self.paths["path_vocab_etype"],
            extractor.vocab_etype,
            write_frequency=False
        )
        logger.info("Saved entity type vocabulary to %s" % self.paths["path_vocab_etype"])

        ##################
        # Training-and-validation loops
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

                for doc_i in train_doc_indices[
                    perm[instance_i: instance_i + batch_size]
                ]:

                    # Forward and compute loss
                    (
                        one_loss,
                        one_acc,
                        n_valid_spans
                    ) = extractor.compute_loss(document=train_documents[doc_i])

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
                        "accuracy": 100.0 * acc_accum / accum_count,
                        "max_valid_f1": bestscore_holder.best_score,
                        "patience": bestscore_holder.patience}
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
                        (extractor.config["n_steps_for_validation"] > 0)
                        and
                        (step % extractor.config["n_steps_for_validation"] == 0)
                    )
                ):

                    ##################
                    # Evaluate
                    ##################

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
                        scores["span_and_type"]["f1"],
                        epoch
                    )

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
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        extractor : BiaffineNER
        documents : list[Document]
        split : str
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # documents -> path_pred
        result_documents = extractor.batch_extract(documents=documents)
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        # (path_pred, path_gold) -> scores
        scores = evaluation.ner.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"]
        )
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
