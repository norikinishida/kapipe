from __future__ import annotations

import copy
import logging
import os
from typing import Any

from spacy.lang.en import English
from tqdm import tqdm

from ..passage_retrieval import BM25, TextSimilarityBasedRetriever
from .. import evaluation
from .. import utils
from ..datatypes import (
    Config,
    Document,
    Mention,
    Entity,
    EntityPassage,
    CandEntKeyInfo,
    CandidateEntitiesForDocument
)


logger = logging.getLogger(__name__)


class Tokenizer:

    def __init__(self):
        self.nlp = English()

    def __call__(self, sentence: str) -> list[str]:
        doc = self.nlp(sentence)
        return [token.text.lower() for token in doc]


class LexicalEntityRetriever:

    def __init__(
        self,
        config: Config | str | None = None,
        path_entity_dict: str | None = None,
        path_snapshot: str | None = None
    ):
        logger.info("########## LexicalEntityRetriever Initialization Starts ##########")

        if path_snapshot is not None:
            assert config is None
            assert path_entity_dict is None
            config = path_snapshot + "/config"
            path_entity_dict = path_snapshot + "/entity_dict.json"

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
 
        # Initialize the tokenizer
        self.tokenizer = Tokenizer()

        # Initialize the retriever
        if self.config["retriever_name"] == "bm25":
            self.retriever = BM25(
                tokenizer=self.tokenizer,
                k1=self.config["k1"],
                b=self.config["b"]
            )
        elif self.config["retriever_name"] == "levenshtein":
            self.retriever = TextSimilarityBasedRetriever(
                normalizer=lambda x: x.lower(),
                similarity_measure="levenshtein"
            )
        else:
            raise Exception(f"Invalid retriever_name: {self.config['retriever_name']}")

        # Create entity passages
        # We expand the entities using synonyms
        # Thus, the number of entity passages >= the number of entities
        entity_passages: list[EntityPassage] = []
        use_desc = "description" in self.config["features"]
        for eid, epage in self.entity_dict.items():
            names = [epage["canonical_name"]] + epage["synonyms"]
            description = epage["description"]
            for name in names:
                entity_passage: EntityPassage = {
                    "id": eid,
                    "title": name,
                    "text": description if use_desc else ""
                }
                entity_passages.append(entity_passage)
        logger.info(f"Number of entities: {len(self.entity_dict)}")
        logger.info(f"Number of entity passages (after synonym expansion): {len(entity_passages)}")

        # Build index
        logger.info("Building index ...")
        self.retriever.make_index(passages=entity_passages)
        logger.info("Completed indexing")

        logger.info("########## LexicalEntityRetriever Initialization Ends ##########")

    def save(self, path_snapshot: str) -> None:
        path_config = path_snapshot + "/config"
        path_entity_dict = path_snapshot + "/entity_dict.json"
        utils.write_json(path_config, self.config)
        utils.write_json(path_entity_dict, self.entity_dict)

    def extract(
        self,
        document: Document,
        retrieval_size: int = 1
    ) -> tuple[Document, CandidateEntitiesForDocument]:
        words = " ".join(document["sentences"]).split()
        mention_pred_entity_ids = [] # (n_mentions, retrieval_size)
        mention_pred_entity_names = [] # (n_mentions, retrieval_size)
        retrieval_scores = [] # (n_mentions, retrieval_size)
        for mention in document["mentions"]:
            # Get query
            begin_i, end_i = mention["span"]
            query = " ".join(words[begin_i : end_i + 1])
            # Retrieval
            entity_passages: list[EntityPassage] = self.retriever.search(
                query=query,
                top_k=self.config["retrieval_size"]
            )
            pred_entity_ids = [p["id"] for p in entity_passages]
            pred_entity_names = [p["title"] for p in entity_passages]
            scores = [p["score"] for p in entity_passages]

            if len(pred_entity_ids) < retrieval_size:
                last_id = pred_entity_ids[-1]
                last_name = pred_entity_names[-1]
                last_score = scores[-1]
                pred_entity_ids = pred_entity_ids + [last_id] * (retrieval_size - len(pred_entity_ids))
                pred_entity_names = pred_entity_names + [last_name] * (retrieval_size - len(pred_entity_names))
                scores = scores + [last_score] * (retrieval_size - len(scores))

            mention_pred_entity_ids.append(pred_entity_ids)
            mention_pred_entity_names.append(pred_entity_names)
            retrieval_scores.append(scores)

        # Structurize (1)
        # Get outputs (mention-level)
        mentions: list[Mention] = []
        for m_i in range(len(document["mentions"])):
            mentions.append({
                "entity_id": mention_pred_entity_ids[m_i][0],
            })

        # Structurize (2)
        # Get outputs (entity-level)
        # i.e., aggregate mentions based on the entity IDs
        entities: list[Entity] = utils.aggregate_mentions_to_entities(
            document=document,
            mentions=mentions
        )

        # Structurize (3)
        # Get outputs (candidate entities for each mention)
        candidate_entities_for_mentions: list[list[CandEntKeyInfo]] = []
        n_mentions = len(mention_pred_entity_ids)
        assert len(mention_pred_entity_ids[0]) == self.config["retrieval_size"]
        for m_i in range(n_mentions):
            lst_cand_ent: list[CandEntKeyInfo] = []
            for c_i in range(self.config["retrieval_size"]):
                cand_ent = {
                    "entity_id": mention_pred_entity_ids[m_i][c_i],
                    "canonical_name": mention_pred_entity_names[m_i][c_i],
                    "score": float(retrieval_scores[m_i][c_i]),
                }
                lst_cand_ent.append(cand_ent)
            candidate_entities_for_mentions.append(lst_cand_ent)

        # Integrate
        document = copy.deepcopy(document)
        for m_i in range(len(document["mentions"])):
            document["mentions"][m_i].update(mentions[m_i])
        document["entities"] = entities
        candidate_entities_for_doc: CandidateEntitiesForDocument = {
            "doc_key": document["doc_key"],
            "candidate_entities": candidate_entities_for_mentions
        }
        return document, candidate_entities_for_doc

    def batch_extract(
        self,
        documents: list[Document],
        retrieval_size: int = 1
    ) -> tuple[list[Document], list[CandidateEntitiesForDocument]]:
        result_documents = []
        candidate_entities = []
        for document in tqdm(documents, desc="extraction steps"):
            document, candidate_entities_for_doc = self.extract(
                document=document,
                retrieval_size=retrieval_size
            )
            result_documents.append(document)
            candidate_entities.append(candidate_entities_for_doc)
        return result_documents, candidate_entities


class LexicalEntityRetrieverTrainer:

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str, Any]:
        paths = {}

        # configurations
        paths["path_snapshot"] = self.base_output_path

        # evaluation outputs
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_pred_retrieval"] = self.base_output_path + "/dev.pred_candidate_entities.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_pred_retrieval"] = self.base_output_path + "/test.pred_candidate_entities.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"
        paths["path_train_pred"] = self.base_output_path + "/train.pred.json"
        paths["path_train_pred_retrieval"] = self.base_output_path + "/train.pred_candidate_entities.json"

        return paths

    def setup_dataset(
        self,
        extractor: LexicalEntityRetriever,
        documents: list[Document],
        split: str
    ) -> None:
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(extractor.entity_dict.keys()))
            gold_documents = []
            for document in tqdm(documents, desc="dataset setup"):
                gold_doc = copy.deepcopy(document)
                for m_i, mention in enumerate(document["mentions"]):
                    in_kb = mention["entity_id"] in kb_entity_ids
                    gold_doc["mentions"][m_i]["in_kb"] = in_kb
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def save_extractor(self, extractor: LexicalEntityRetriever) -> None:
        extractor.save(path_snapshot=self.paths["path_snapshot"])
        logger.info(f"Saved config and entity dictionary to {self.paths['path_snapshot']}")

    def evaluate(
        self,
        extractor: LexicalEntityRetriever,
        documents: list[Document],
        split: str,
        #
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:
        # Apply the extractor
        result_documents, candidate_entities = extractor.batch_extract(
            documents=documents,
            retrieval_size=extractor.config["retrieval_size"]
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        utils.write_json(
            self.paths[f"path_{split}_pred_retrieval"],
            candidate_entities
        )

        if prediction_only:
            return

        # Calculate the evaluation scores
        scores = evaluation.ed.accuracy(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True,
            skip_normalization=True
        )
        scores.update(evaluation.ed.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True,
            skip_normalization=True
        ))
        scores.update(evaluation.ed.recall_at_k(
            pred_path=self.paths[f"path_{split}_pred_retrieval"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        ))

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
