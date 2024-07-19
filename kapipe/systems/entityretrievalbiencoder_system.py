import copy
import logging

import numpy as np
import torch
from tqdm import tqdm

from ..models import EntityRetrievalBiEncoderModel
from ..misc import ApproximateNearestNeighborSearch
from .. import utils


logger = logging.getLogger(__name__)


class EntityRetrievalBiEncoderSystem:
    """Bi-Encoder in BLINK (Wu et al., 2020).
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
        path_entity_dict : str
        path_model: str | None
            by default None
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> BiEncoderSystem Initialization >>>>>>>>>>")
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
        # Approximate Nearest Neighbor Search
        ######

        self.anns = ApproximateNearestNeighborSearch()
        self.index_made = False

        ######
        # Model
        ######

        self.model_name = config["model_name"]

        self.special_entity_sep_marker = ":"

        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        if self.verbose:
            logger.info(f"Loaded entity dictionary from {path_entity_dict}")

        if self.model_name == "entityretrievalbiencodermodel":
            self.model = EntityRetrievalBiEncoderModel(
                device=device,
                bert_pretrained_name_or_path=config[
                    "bert_pretrained_name_or_path"
                ],
                max_seg_len=config["max_seg_len"]
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
            logger.info("<<<<<<<<<< BiEncoderSystem Initialization <<<<<<<<<<")

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
        self.precomputed_entity_vectors = np.load(
            path.replace("/model", "/entity_vectors.npy")
        )

    def save_model(self, path):
        """
        Parameters
        ----------
        path : str
        """
        torch.save(self.model.state_dict(), path)
        np.save(
            path.replace("/model", "/entity_vectors.npy"),
            self.precomputed_entity_vectors
        )

    #####
    # For training
    #####

    def compute_loss(
        self,
        document,
        candidate_entities_for_doc,
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, list[CandEntKeyInfo]]

        Returns
        -------
        tuple[torch.Tensor, int]
        """
        # Switch to training mode
        self.model.train()

        # Generate entity documents
        candidate_entity_documents = []
        for cand in candidate_entities_for_doc["candidate_entities"]:
            entity_id = cand["entity_id"]
            epage = self.entity_dict[entity_id]
            canonical_name = epage["canonical_name"]
            # synonyms = epage["synonyms"]
            description = epage["description"]
            edoc = {
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "text": " ".join([
                    canonical_name,
                    self.special_entity_sep_marker,
                    description
                ])
            }
            candidate_entity_documents.append(edoc)

        # Preprocess, tensorize, and encode entities
        # (n_candidates, hidden_dim)
        candidate_entity_vectors = self.preprocess_and_encode_entities(
            candidate_entity_documents=candidate_entity_documents,
            compute_loss=True
        )

        # Preprocess mentions
        preprocessed_data_m = self.model.preprocess_mentions(
            document=document,
            candidate_entity_documents=candidate_entity_documents
        )

        # Tensorize mentions
        model_input_m = self.model.tensorize_mentions(
            preprocessed_data=preprocessed_data_m,
            compute_loss=True
        )

        # Encode mentions
        # (n_mentions, hidden_dim)
        mention_vectors = self.model.encode_mentions(**model_input_m)

        # Compute scores
        model_input = self.model.tensorize(
            preprocessed_data=preprocessed_data_m,
            compute_loss=True
        )
        model_output = self.model.forward(
            mention_vectors=mention_vectors,
            candidate_entity_vectors=candidate_entity_vectors,
            **model_input
        )

        return (
            model_output.loss,
            model_output.n_mentions
        )

    #####
    # For inference
    #####

    def make_index(self, use_precomputed_entity_vectors=False):
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Generate entity documents
            entity_documents = []
            for entity_id, epage in self.entity_dict.items():
                canonical_name = epage["canonical_name"]
                # synonyms = epage["synonyms"]
                description = epage["description"]
                edoc = {
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "text": " ".join([
                        canonical_name,
                        self.special_entity_sep_marker,
                        description
                    ])
                }
                entity_documents.append(edoc)

            # Preprocess, tensorize, and encode entities
            if use_precomputed_entity_vectors:
                all_entity_vectors = self.precomputed_entity_vectors
            else:
                all_entity_vectors = self.preprocess_and_encode_entities(
                    candidate_entity_documents=entity_documents,
                    compute_loss=False
                )

            # Make ANNS index
            self.anns.make_index(
                document_vectors=all_entity_vectors,
                document_ids=[edoc["entity_id"] for edoc in entity_documents],
                document_titles=[
                    edoc["canonical_name"] for edoc in entity_documents
                ]
            )

            self.precomputed_entity_vectors = all_entity_vectors
            self.index_made = True

    def extract(self, document, retrieval_size=1):
        """
        Parameters
        ----------
        document : Document
        retrieval_size: int
            by default 1

        Returns
        -------
        tuple[Document, dict[str, str | list[list[CandEntKeyInfo]]]]
        """
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Preprocess mentions
            preprocessed_data_m = self.model.preprocess_mentions(
                document=document,
                candidate_entity_documents=None
            )

            # Tensorize mentions
            model_input_m = self.model.tensorize_mentions(
                preprocessed_data=preprocessed_data_m,
                compute_loss=False
            )

            # Encode mentions
            # (n_mentions, hidden_dim)
            mention_vectors = self.model.encode_mentions(**model_input_m)

            # Approximate Nearest Neighbor Search
            # (n_mentions, retrieval_size), (n_mentions, retrieval_size),
            #   (n_mentions, retrieval_size)
            (
                mention_pred_entity_ids,
                mention_pred_entity_names,
                retrieval_scores
            ) = self.anns.search(
                query_vectors=mention_vectors.cpu().numpy(),
                retrieval_size=retrieval_size
            )

            # Structurize (1)
            # Transform to mention-level entity IDs
            mentions = [] # list[Mention]
            for m_i in range(len(preprocessed_data_m["mentions"])):
                mentions.append({
                    "entity_id": mention_pred_entity_ids[m_i][0],
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

            # Structuriaze (3)
            # Transform to candidate entities for each mention
            candidate_entities_for_mentions = [] # list[list[CandEntKeyInfo]]
            n_mentions = len(mention_pred_entity_ids)
            assert len(mention_pred_entity_ids[0]) == retrieval_size
            for m_i in range(n_mentions):
                lst_cand_ent = [] # list[CandEntKeyInfo]
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

    def batch_extract(self, documents, retrieval_size=1):
        """
        Parameters
        ----------
        documents : list[Document]
        retrieval_size : int
            by default 1

        Returns
        -------
        tuple[list[Document], list[dict[str, str | list[list[CandEntKeyInfo]]]]]
        """
        result_documents = []
        candidate_entities = []
        for document in tqdm(documents, desc="extraction steps"):
            result_document, candidate_entities_for_doc \
                = self.extract(
                    document=document,
                    retrieval_size=retrieval_size
                )
            result_documents.append(result_document)
            candidate_entities.append(candidate_entities_for_doc)
        return result_documents, candidate_entities

    #####
    # Subfunctions
    #####

    def preprocess_and_encode_entities(
        self,
        candidate_entity_documents,
        compute_loss
    ):
        """
        Parameters
        ----------
        candidate_entity_documents : list[EntDoc]
        compute_loss : bool

        Returns
        -------
        torch.Tensor | numpy.ndarray
            shape of (n_candidates, hidden_dim)
        """
        BATCH_SIZE = 5

        candidate_entity_vectors = []

        if compute_loss:
            generator = range(0, len(candidate_entity_documents), BATCH_SIZE)
        else:
            generator = tqdm(
                range(0, len(candidate_entity_documents), BATCH_SIZE),
                desc="entity encoding"
            )

        for e_i in generator:
            # Generate batch entity documents
            batch_entity_documents \
                = candidate_entity_documents[e_i : e_i + BATCH_SIZE]
            # Preprocess entities
            preprocessed_data_e = self.model.preprocess_entities(
                candidate_entity_documents=batch_entity_documents
            )
            # Tensorize entities
            model_input_e = self.model.tensorize_entities(
                preprocessed_data=preprocessed_data_e,
                compute_loss=compute_loss
            )
            # Encode entities
            # (BATCH_SIZE, hidden_dim)
            batch_entity_vectors = self.model.encode_entities(**model_input_e)
            if not compute_loss:
                batch_entity_vectors = batch_entity_vectors.cpu().numpy()
            candidate_entity_vectors.append(batch_entity_vectors)
        # (n_candidates, hidden_dim)
        if compute_loss:
            candidate_entity_vectors = torch.cat(candidate_entity_vectors, dim=0)
        else:
            candidate_entity_vectors = np.concatenate(candidate_entity_vectors, axis=0)
        return candidate_entity_vectors

