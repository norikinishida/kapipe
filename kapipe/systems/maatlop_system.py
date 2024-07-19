import copy
import logging

import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm

from ..models import MAATLOPModel
from .. import utils


logger = logging.getLogger(__name__)


class MAATLOPSystem:
    """Mention-Agnostic ATLOP (Oumaima and Nishida et al., 2024)
    """

    def __init__(
        self,
        device,
        config,
        path_entity_dict,
        vocab_relation,
        path_model=None,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        path_entity_dict : str
        vocab_relation: dict[str, int] | str
        path_model: str | None
            by default None
        verbose: bool | None
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> MAATLOPSystem Initialization >>>>>>>>>>")
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

        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        self.kb_entity_ids = list(self.entity_dict.keys())
        if self.verbose:
            logger.info(f"Loaded entity dictionary from {path_entity_dict}")
 
        if self.model_name == "maatlopmodel":
            self.model = MAATLOPModel(
                device=device,
                bert_pretrained_name_or_path=config[
                    "bert_pretrained_name_or_path"
                ],
                max_seg_len=config["max_seg_len"],
                entity_dict=self.entity_dict,
                entity_seq_length=config["entity_seq_length"],
                use_localized_context_pooling=config[
                    "use_localized_context_pooling"
                ],
                bilinear_block_size=config["bilinear_block_size"],
                use_entity_loss=self.config["do_negative_entity_sampling"],
                vocab_relation=self.vocab_relation,
                possible_head_entity_types=config["possible_head_entity_types"],
                possible_tail_entity_types=config["possible_tail_entity_types"],
                use_mention_as_canonical_name=config[
                    "use_mention_as_canonical_name"
                ]
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
            logger.info("<<<<<<<<<< MAATLOPSystem Initialization <<<<<<<<<<")

    def load_model(self, path, ignored_names=None):
        """
        Parameters
        ----------
        path : str
        ignored_names : list[str] | None
            by default None
        """
        if ignored_names is None:
            self.model.load_state_dict(
                torch.load(path, map_location=torch.device("cpu")),
                strict=False
            )
        else:
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
            for name in ignored_names:
                logger.info(f"Ignored {name} module in loading")
                checkpoint = {
                    k:v for k,v in checkpoint.items() if not name in k
                }
            self.model.load_state_dict(checkpoint, strict=False)

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
        tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor, int] | tuple[torch.Tensor, torch.Tensor, int, int]
        """
        # Switch to training mode
        self.model.train()

        # Negative Entity Sampling
        if self.config["do_negative_entity_sampling"]:
            document = self.sample_negative_entities_randomly(
                document=document,
                sample_size=round(
                    len(document["entities"])
                    * self.config["negative_entity_ratio"]
                )
            )

        # Preprocess
        preprocessed_data = self.model.preprocess(document=document)

        # Tensorize
        model_input = self.model.tensorize(
            preprocessed_data=preprocessed_data,
            compute_loss=True
        )

        # Forward
        model_output = self.model.forward(**model_input)

        if self.config["do_negative_entity_sampling"]:
            return (
                model_output.pair_loss,
                model_output.pair_acc,
                model_output.n_valid_pairs,
                model_output.n_valid_triples,
                #
                model_output.entity_loss,
                model_output.n_entities,
            )
        else:
            return (
                model_output.pair_loss,
                model_output.pair_acc,
                model_output.n_valid_pairs,
                model_output.n_valid_triples
            )

    def sample_negative_entities_randomly(
        self,
        document,
        sample_size
    ):
        """
        Parameters
        ----------
        document : Document
        sample_size : int

        Returns
        -------
        Document
        """
        result_document = copy.deepcopy(document)

        n_entities = len(result_document["entities"])

        gold_entity_ids = [e["entity_id"] for e in result_document["entities"]]

        # Sample candidate entity ids from the entire KB
        sampled_entity_ids = np.random.choice(
            self.kb_entity_ids,
            sample_size + len(gold_entity_ids),
            replace=False
        )

        # Remove gold entities from the sampled list
        sampled_entity_ids = [
            eid for eid in sampled_entity_ids if not eid in gold_entity_ids
        ]
        sampled_entity_ids = sampled_entity_ids[:sample_size]

        # Retrieve the names and types for the sampled entity ids from the KB
        sampled_entity_names = []
        sampled_entity_types = []
        for eid in sampled_entity_ids:
            epage = self.entity_dict[eid]
            name = epage["canonical_name"]
            etype = epage["entity_type"]
            sampled_entity_names.append(name)
            sampled_entity_types.append(etype)

        # Integrate the sampled entities to the document
        sampled_entity_mention_index = []
        for name, etype, eid in zip(
            sampled_entity_names,
            sampled_entity_types,
            sampled_entity_ids
        ):
            mention = {
                "span": None,
                "name": name,
                "entity_type": etype,
                "entity_id": eid,
            }
            result_document["mentions"].append(mention)
            sampled_entity_mention_index.append(len(result_document["mentions"]) - 1)

        for e_i in range(len(result_document["entities"])):
            result_document["entities"][e_i]["is_dummy"] = True

        for m_i, etype, eid in zip(
            sampled_entity_mention_index,
            sampled_entity_types,
            sampled_entity_ids
        ):
            entity = {
                "mention_indices": [m_i],
                "entity_type": etype,
                "entity_id": eid,
                "is_dummy": False,
            }
            result_document["entities"].append(entity)

        assert len(result_document["entities"]) == n_entities + sample_size
        return result_document

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
            logits = model_output.pair_logits

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
        pair_pred_relation_labels \
            = self.model.pair_loss_function.get_labels(
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

