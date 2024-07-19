import copy
import logging

import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm

from ..models import ATLOPModel
from .. import utils


logger = logging.getLogger(__name__)


class ATLOPSystem:
    """ATLOP (Zhou et al., 2021).
    """

    def __init__(
        self,
        device,
        config,
        vocab_relation,
        path_model=None,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        vocab_relation: dict[str, int] | str
        path_model: str | None
            by default None
        verbose: bool | None
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> ATLOPSystem Initialization >>>>>>>>>>")
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
        # Vocabulary (relations)
        ######

        if isinstance(vocab_relation, str):
            tmp = vocab_relation
            vocab_relation = utils.read_vocab(vocab_relation)
            if self.verbose:
                logger.info(f"Loaded relation type vocabulary from {tmp}")
        self.vocab_relation = vocab_relation
        self.ivocab_relation = {i:l for l, i in self.vocab_relation.items()}

        ######
        # Model
        ######

        self.model_name = config["model_name"]
        self.top_k_labels = config["top_k_labels"]
        if self.model_name == "atlopmodel":
            self.model = ATLOPModel(
                device=device,
                bert_pretrained_name_or_path=config[
                    "bert_pretrained_name_or_path"
                ],
                max_seg_len=config["max_seg_len"],
                token_embedding_method=config["token_embedding_method"],
                entity_pooling_method=config["entity_pooling_method"],
                use_localized_context_pooling=config[
                    "use_localized_context_pooling"
                ],
                bilinear_block_size=config["bilinear_block_size"],
                vocab_relation=self.vocab_relation,
                loss_function_name=config["loss_function"],
                possible_head_entity_types=config["possible_head_entity_types"],
                possible_tail_entity_types=config["possible_tail_entity_types"]
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

        if self.verbose:
            logger.info("<<<<<<<<<< ATLOPSystem Initialization <<<<<<<<<<")

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
        tuple[torch.Tensor, torch.Tensor, int, int]
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
            model_output.n_valid_pairs,
            model_output.n_valid_triples
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
            if (
                len(preprocessed_data["pair_head_entity_indices"]) == 0
                or
                len(preprocessed_data["pair_tail_entity_indices"]) == 0
            ):
                result_document = copy.deepcopy(document)
                result_document["relations"] = []
                return result_document

            # Tensorize
            model_input = self.model.tensorize(
                preprocessed_data=preprocessed_data,
                compute_loss=False
            )

            # Forward
            model_output = self.model.forward(**model_input)
            # (n_entity_pairs, n_relations)
            logits = model_output.logits

            # Structurize
            triples = self.structurize(
                pair_head_entity_indices=preprocessed_data["pair_head_entity_indices"],
                pair_tail_entity_indices=preprocessed_data["pair_tail_entity_indices"],
                logits=logits
            )

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["relations"] = triples
            return result_document

    def structurize(
        self,
        pair_head_entity_indices,
        pair_tail_entity_indices,
        logits
    ):
        triples = [] # list[Triple]

        # (n_entity_pairs, n_relations)
        pair_pred_relation_labels = self.model.loss_function.get_labels(
            logits=logits,
            top_k=self.top_k_labels
        ).cpu().numpy()

        for head_entity_i, tail_entity_i, rel_indicators in zip(
            pair_head_entity_indices,
            pair_tail_entity_indices,
            pair_pred_relation_labels
        ):
            assert head_entity_i != tail_entity_i
            # Find non-zero relation IDs (indices)
            rel_indices = np.nonzero(rel_indicators)[0].tolist()
            for rel_i in rel_indices:
                if rel_i != 0:
                    rel = self.ivocab_relation[rel_i]
                    triples.append({
                        "arg1": int(head_entity_i),
                        "relation": rel,
                        "arg2": int(tail_entity_i),
                        })

        return triples

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

