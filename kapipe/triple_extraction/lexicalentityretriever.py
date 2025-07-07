import copy
import logging
import os

from spacy.lang.en import English
from tqdm import tqdm

from ..passage_retrieval import BM25, TextSimilarityBasedRetriever
from .. import evaluation
from .. import utils


logger = logging.getLogger(__name__)


class Tokenizer:

    def __init__(self):
        self.nlp = English()

    def __call__(self, sentence):
        doc = self.nlp(sentence)
        return [token.text.lower() for token in doc]


class LexicalEntityRetriever:

    def __init__(
        self,
        # General
        config,
        # Task specific
        path_entity_dict,
        # Misc.
        verbose=True
    ):
        """
        Parameters
        ----------
        config : ConfigTree | str
        path_entity_dict : str
        verbose : bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> LexicalEntityRetriever Initialization >>>>>>>>>>")

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
        # Retriever
        ######

        self.tokenizer = Tokenizer()

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

        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        if self.verbose:
            logger.info(f"Loaded entity dictionary from {path_entity_dict}")

        # Make index
        logger.info("Making index ...")

        # def get_text(name, description):
        #     text = []
        #     if "canonical_name" in self.config["features"]:
        #         text.append(name)
        #     if "description" in self.config["features"]:
        #         text.append(description)
        #     return " ".join(text)
                
        # Generate entity passages
        # We expand the entities using synonyms
        # Thus, the number of entity passages >= the number of entities
        entity_passages = [] # list[EntityPassage]      
        use_desc = "description" in self.config["features"]
        for eid, epage in self.entity_dict.items():
            # desc = epage["description"]
            names = [epage["canonical_name"]] + epage["synonyms"]
            description = epage["description"]
            for name in names:
                entity_passage = {
                    "id": eid,
                    "title": name,
                    # "text": get_text(name, epage["description"]),
                    "text": description if use_desc else ""
                }
                entity_passages.append(entity_passage)
        logger.info(f"Number of entities: {len(self.entity_dict)}")
        logger.info(f"Number of entity passages (after synonym expansion): {len(entity_passages)}")

        self.retriever.make_index(passages=entity_passages)
        logger.info("Made index")

        if self.verbose:
            logger.info("<<<<<<<<<< LexicalEntityRetriever Initialization <<<<<<<<<<")

    # ---

    def extract(self, document, retrieval_size=1):
        """
        Parameters
        ----------
        document : Document
        retrieval_size : int
            by default 1

        Returns
        -------
        tuple[Document, dict[str, list[list[CandEntKeyInfo]]]]
        """
        words = " ".join(document["sentences"]).split()
        mention_pred_entity_ids = [] # (n_mentions, retrieval_size)
        mention_pred_entity_names = [] # (n_mentions, retrieval_size)
        retrieval_scores = [] # (n_mentions, retrieval_size)
        for mention in document["mentions"]:
            # Get query
            begin_i, end_i = mention["span"]
            query = " ".join(words[begin_i : end_i + 1])
            # Retrieval
            # pred_entity_ids, pred_entity_names, scores =
            entity_passages = self.retriever.search(
                query=query,
                top_k=self.config["retrieval_size"]
            ) # list[Passage]
            pred_entity_ids = [p["id"] for p in entity_passages]
            pred_entity_names = [p["title"] for p in entity_passages]
            scores = [p["score"] for p in entity_passages]
            mention_pred_entity_ids.append(pred_entity_ids)
            mention_pred_entity_names.append(pred_entity_names)
            retrieval_scores.append(scores)

        # Structurize (1)
        # Get outputs (mention-level)
        mentions = [] # list[Mention]
        for m_i in range(len(document["mentions"])):
            mentions.append({
                "entity_id": mention_pred_entity_ids[m_i][0],
            })

        # Structurize (2)
        # Get outputs (entity-level)
        # i.e., aggregate mentions based on the entity IDs
        entities = utils.aggregate_mentions_to_entities(
            document=document,
            mentions=mentions
        )

        # Structurize (3)
        # Get outputs (candidate entities for each mention)
        candidate_entities_for_mentions = [] # list[list[CandEntKeyInfo]]
        n_mentions = len(mention_pred_entity_ids)
        assert len(mention_pred_entity_ids[0]) == self.config["retrieval_size"]
        for m_i in range(n_mentions):
            lst_cand_ent = [] # list[CandEntKeyInfo]
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
        candidate_entities_for_doc = {
            "doc_key": document["doc_key"],
            "candidate_entities": candidate_entities_for_mentions
        }
        return document, candidate_entities_for_doc

    def batch_extract(self, documents, retrieval_size=1):
        """
        Parameters
        ----------
        documents : list[Document]
        retrieval_size : int
            by default 1

        Returns
        -------
        tuple[list[Document], list[dict[str, list[list[CandEntKeyInfo]]]]]
        """
        result_documents = []
        candidate_entities = []
        for document in tqdm(documents, desc="extraction steps"):
            document, candidate_entities_for_doc \
                = self.extract(
                    document=document,
                    retrieval_size=retrieval_size
                )
            result_documents.append(document)
            candidate_entities.append(candidate_entities_for_doc)
        return result_documents, candidate_entities


class LexicalEntityRetrieverTrainer:

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

        # Paths to validation outputs and scores
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_pred_retrieval"] = self.base_output_path + "/dev.pred_candidate_entities.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"

        # Paths to evaluation outputs and scores
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_pred_retrieval"] = self.base_output_path + "/test.pred_candidate_entities.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        paths["path_train_pred"] = self.base_output_path + "/train.pred.json"
        paths["path_train_pred_retrieval"] = self.base_output_path + "/train.pred_candidate_entities.json"

        return paths

    def setup_dataset(self, extractor, documents, split):
        """
        Parameters
        ----------
        extractor : LexicalEntityRetriever
        documents : list[Document]
        split : str
        """
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

    def save_extractor(self, extractor):
        """
        Parameters
        ----------
        extractor : LexicalizedEntityRetriever
        """
        # utils.dump_hocon_config(self.paths["path_config"], extractor.config)
        utils.write_json(self.paths["path_config"], extractor.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

    def evaluate(
        self,
        extractor,
        documents,
        split,
        #
        prediction_only=False,
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        extractor : LexicalEntityRetriever
        documents : list[Document]
        split : str
        prediction_only : bool
            by default False
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any] | None
        """
        # (documents, entity_dict) -> path_pred
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
        # (path_pred, path_gold) -> scores
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
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
