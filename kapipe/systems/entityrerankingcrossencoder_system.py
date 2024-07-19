import copy
import logging

# import numpy as np
import torch
from tqdm import tqdm

from ..models import EntityRerankingCrossEncoderModel
from .. import utils


logger = logging.getLogger(__name__)


class EntityRerankingCrossEncoderSystem:
    """Cross-Encoder in BLINK (Wu et al., 2020).
    """

    def __init__(
        self,
        device,
        config,
        path_entity_dict,
        path_model=None,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        path_entity_dict: str
        path_model: str | None
            by default None
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> CrossEncoderSystem Initialization >>>>>>>>>>")
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
        # Model
        ######

        self.model_name = config["model_name"]

        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        if self.verbose:
            logger.info(f"Loaded entity dictionary from {path_entity_dict}")

        if self.model_name == "entityrerankingcrossencodermodel":
            self.model = EntityRerankingCrossEncoderModel(
                device=device,
                bert_pretrained_name_or_path=config[
                    "bert_pretrained_name_or_path"
                ],
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
        if path_model is not None:
            self.load_model(path=path_model)
            if self.verbose:
                logger.info(f"Loaded model parameters from {path_model}")

        self.model.to(self.model.device)

        if self.verbose:
            logger.info("<<<<<<<<<< CrossEncoderSystem Initialization <<<<<<<<<<")

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

    def compute_loss(
        self,
        document,
        candidate_entities_for_doc,
        mention_index
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, str | list[list[CandEntKeyInfo]]]
        mention_index : int

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
        """
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

    def extract(self, document, candidate_entities_for_doc):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, str | list[list[CandEntKeyInfo]]]

        Returns
        -------
        Document
        """
        assert document["doc_key"] == candidate_entities_for_doc["doc_key"]

        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Preprocess
            preprocessed_data = self.model.preprocess(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc,
                max_n_candidates=self.config["max_n_candidates_in_inference"]
            )

            # Get outputs (mention-level) by iterating over the mentions
            mentions = [] # list[Mention]
            cands_for_mentions \
                = candidate_entities_for_doc["candidate_entities"]
            for mention_index in range(len(preprocessed_data["mentions"])):
                # Tensorize
                model_input = self.model.tensorize(
                    preprocessed_data=preprocessed_data,
                    mention_index=mention_index,
                    compute_loss=False
                )

                # Forward
                model_output = self.model.forward(**model_input)
                # (1, n_candidates)
                logits = model_output.logits

                # Structurize (1)
                # Transform logits to mention-level entity IDs
                pred_candidate_entity_index \
                    = torch.argmax(logits, dim=1).cpu().item() # int
                pred_candidate_entity_id = (
                    cands_for_mentions
                    [mention_index]
                    [pred_candidate_entity_index]
                    ["entity_id"]
                )
                mentions.append({
                    "entity_id": pred_candidate_entity_id,
                })

            # Structurize (2)
            # Transform to entity-level entity IDs
            # i.e., aggregate mentions based on the entity IDs
            entity_id_to_info = {} # dict[str, dict[str, Any]]
            for m_i in range(len(document["mentions"])):
                entity_type = document["mentions"][m_i]["entity_type"]
                entity_id = mentions[m_i]["entity_id"]
                if entity_id in entity_id_to_info:
                    entity_id_to_info[entity_id]["mention_indices"].append(m_i)
                    # TODO
                    # Confliction of entity types can appear, if EL model does not care about it.
                    # assert (
                    #     entity_id_to_info[entity_id]["entity_type"]
                    #     == entity_type
                    # )
                else:
                    entity_id_to_info[entity_id] = {}
                    entity_id_to_info[entity_id]["mention_indices"] = [m_i]
                    # TODO
                    entity_id_to_info[entity_id]["entity_type"] = entity_type
            entities = [] # list[Entity]
            for entity_id in entity_id_to_info.keys():
                mention_indices \
                    = entity_id_to_info[entity_id]["mention_indices"]
                entity_type = entity_id_to_info[entity_id]["entity_type"]
                entities.append({
                    "mention_indices": mention_indices,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                })

            # Integrate
            result_document = copy.deepcopy(document)
            for m_i in range(len(result_document["mentions"])):
                result_document["mentions"][m_i].update(mentions[m_i])
            result_document["entities"] = entities
            return result_document

    def batch_extract(self, documents, candidate_entities):
        """
        Parameters
        ----------
        documents : list[Document]
        candidate_entities : list[dict[str, str | list[list[CandEntKeyInfo]]]]

        Returns
        -------
        list[Document
        """
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
