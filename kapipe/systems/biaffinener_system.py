import copy
import logging

import numpy as np
import torch
from tqdm import tqdm

from ..models import BiaffineNERModel
from ..decoders import SpanBasedDecoder
from .. import utils


logger = logging.getLogger(__name__)


class BiaffineNERSystem:
    """ Biaffine-NER (Yu et al., 2020).
    """

    def __init__(
        self,
        device,
        config,
        vocab_etype,
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
            logger.info(">>>>>>>>>> BiaffineNERSystem Initialization >>>>>>>>>>")
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
            logger.info("<<<<<<<<<< BiaffineNERSystem Initialization <<<<<<<<<<")

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

