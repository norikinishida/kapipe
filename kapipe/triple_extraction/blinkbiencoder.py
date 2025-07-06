import copy
import logging
import time

import numpy as np
import torch
from tqdm import tqdm

from ..models import BlinkBiEncoderModel
from ..passage_retrieval import ApproximateNearestNeighborSearch
from .. import utils


logger = logging.getLogger(__name__)


class BlinkBiEncoder:
    """Bi-Encoder in BLINK (Wu et al., 2020).
    """

    def __init__(
        self,
        # General
        device,
        config,
        # Task specific
        path_entity_dict,
        # Misc.
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
            logger.info(">>>>>>>>>> BlinkBiEncoder Initialization >>>>>>>>>>")
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

        # TODO: Allow GPU-ID selection
        # NOTE: The GPU ID for indexing should NOT be the same with the GPU ID of the BLINK model to avoid OOM error
        #       Here, we assume that GPU-0 is set for the BLINK model.
        self.anns = ApproximateNearestNeighborSearch(gpu_id=1)

        ######
        # Model
        ######

        self.model_name = config["model_name"]

        # self.special_entity_sep_marker = ":"

        if self.verbose:
            logger.info(f"Loading entity dictionary from {path_entity_dict}")
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        if self.verbose:
            logger.info(f"Completed loading of entity dictionary with {len(self.entity_dict)} entities from {path_entity_dict}")

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

        # Load trained model parameters
        if path_model is not None:
            self.load_model(path=path_model)
            if self.verbose:
                logger.info(f"Loaded model parameters from {path_model}")

        self.model.to(self.model.device)

        if self.verbose:
            logger.info("<<<<<<<<<< BlinkBiEncoder Initialization <<<<<<<<<<")

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
        self.precomputed_entity_vectors = np.load(path.replace("/model", "/entity_vectors.npy"))

    def save_model(self, path):
        """
        Parameters
        ----------
        path : str
        """
        torch.save(self.model.state_dict(), path)
        np.save(path.replace("/model", "/entity_vectors.npy"), self.precomputed_entity_vectors)

    #####
    # For training
    #####

    def compute_loss(
        self,
        document,
        flatten_candidate_entities_for_doc,
    ):
        """
        Parameters
        ----------
        document : Document
        flatten_candidate_entities_for_doc : dict[str, list[CandEntKeyInfo]]

        Returns
        -------
        tuple[torch.Tensor, int]
        """
        # Switch to training mode
        self.model.train()

        ###############
        # Entity Encoding
        ###############

        # Generate entity passages
        candidate_entity_passages = []
        for cand in flatten_candidate_entities_for_doc["flatten_candidate_entities"]:
            entity_id = cand["entity_id"]
            epage = self.entity_dict[entity_id]
            canonical_name = epage["canonical_name"]
            # synonyms = epage["synonyms"]
            description = epage["description"]
            entity_passage = {
                "id": entity_id,
                "title": canonical_name,
                # "text": " ".join([
                #     canonical_name,
                #     self.special_entity_sep_marker,
                #     description
                # ])
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
        preprocessed_data_m = self.model.preprocess_mentions(
            document=document,
        )

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

    #####
    # For inference
    #####

    def make_index(self, use_precomputed_entity_vectors=False):
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()
            start_time = time.time()

            # Generate entity passages
            if self.verbose:
                logger.info(f"Building passages for {len(self.entity_dict)} entities ...")
            entity_passages = []
            for entity_id, epage in self.entity_dict.items():
                canonical_name = epage["canonical_name"]
                # synonyms = epage["synonyms"]
                description = epage["description"]
                entity_passage = {
                    "id": entity_id,
                    "title": canonical_name,
                    # "text": " ".join([
                    #     canonical_name,
                    #     self.special_entity_sep_marker,
                    #     description
                    # ])
                    "text": description,
                }
                entity_passages.append(entity_passage)

            # Preprocess, tensorize, and encode entities
            if use_precomputed_entity_vectors:
                entity_vectors = self.precomputed_entity_vectors
            else:
                if self.verbose:
                    logger.info(f"Encoding {len(entity_passages)} entities ...")
                # entity_vectors = np.random.random((len(entity_passages), 768)).astype(np.float32)
                pool = self.model.start_multi_process_pool()
                entity_vectors = self.model.encode_multi_process(entity_passages, pool)
                self.model.stop_multi_process_pool(pool)
                self.model.to(self.device)

            # Make ANNS index
            if self.verbose:
                logger.info(f"Indexing {len(entity_vectors)} entities ...")
            self.anns.make_index(
                passage_vectors=entity_vectors,
                passage_ids=[p["id"] for p in entity_passages],
                passage_metadatas=[{"title": p["title"]} for p in entity_passages]
            )

            self.precomputed_entity_vectors = entity_vectors

            if self.verbose:
                end_time = time.time()
                span_time = end_time - start_time
                span_time /= 60.0
                logger.info("Completed indexing")
                logger.info(f"Time: {span_time} min.")

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

            if len(document["mentions"]) == 0:
                result_document = copy.deepcopy(document)
                result_document["entities"] = []
                candidate_entities_for_doc = {
                   "doc_key": result_document["doc_key"],
                   "candidate_entities": []
                }
                return result_document, candidate_entities_for_doc

            # Preprocess mentions
            preprocessed_data_m = self.model.preprocess_mentions(
                document=document,
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
            mentions = [] # list[Mention]
            for m_i in range(len(preprocessed_data_m["mentions"])):
                mentions.append({
                    "entity_id": mention_pred_entity_ids[m_i][0],
                })

            # Structurize (2)
            # Transform to entity-level entity IDs
            # i.e., aggregate mentions based on the entity IDs
            entities = utils.aggregate_mentions_to_entities(
                document=document,
                mentions=mentions
            )

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

    # def encode_entities_with_multi_processing(
    #     self,
    #     entity_passages,
    # ):
    #     """
    #     Parameters
    #     ----------
    #     entity_passages: list[EntityPassage]

    #     Returns
    #     -------
    #     numpy.ndarray
    #         shape of (n_candidates, hidden_dim)
    #     """
    #     # if compute_loss:
    #     #     BATCH_SIZE = 4
    #     # else:
    #     #     BATCH_SIZE = 512

    #     # candidate_entity_vectors = []

    #     # if compute_loss:
    #     #     generator = range(0, len(candidate_entity_passages), BATCH_SIZE)
    #     # else:
    #     #     generator = tqdm(
    #     #         range(0, len(candidate_entity_passages), BATCH_SIZE),
    #     #         desc="entity encoding"
    #     #     )

    #     # for e_i in generator:
    #     #     # Generate batch entity passages
    #     #     batch_entity_passages \
    #     #         = candidate_entity_passages[e_i : e_i + BATCH_SIZE]
    #     #     # Preprocess entities
    #     #     preprocessed_data_e = self.model.preprocess_entities(
    #     #         candidate_entity_passages=batch_entity_passages
    #     #     )
    #     #     # Tensorize entities
    #     #     model_input_e = self.model.tensorize_entities(
    #     #         preprocessed_data=preprocessed_data_e,
    #     #         compute_loss=compute_loss
    #     #     )
    #     #     # Encode entities
    #     #     # (BATCH_SIZE, hidden_dim)
    #     #     batch_entity_vectors = self.model.encode_entities(**model_input_e)
    #     #     if not compute_loss:
    #     #         # batch_entity_vectors = batch_entity_vectors.cpu().numpy()
    #     #         batch_entity_vectors = batch_entity_vectors.cpu()
    #     #     candidate_entity_vectors.append(batch_entity_vectors)
    #     # # (n_candidates, hidden_dim)
    #     # if compute_loss:
    #     #     candidate_entity_vectors = torch.cat(candidate_entity_vectors, dim=0)
    #     # else:
    #     #     # candidate_entity_vectors = np.concatenate(candidate_entity_vectors, axis=0)
    #     #     candidate_entity_vectors = torch.cat(candidate_entity_vectors, dim=0).numpy()
    #     # return candidate_entity_vectors

    #     pool = self.model.start_multi_process_pool()

    #     entity_vectors = self.model.encode_multi_process(entity_passages, pool)

    #     self.model.stop_multi_process_pool(pool)
    #     self.model.to(self.device)

    #     return entity_vectors
