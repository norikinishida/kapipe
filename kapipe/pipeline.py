import logging
import os
from os.path import expanduser

from .systems import BiaffineNERSystem
from .systems import EntityRetrievalBiEncoderSystem
from .systems import EntityRerankingCrossEncoderSystem
from .systems import ATLOPSystem

from .trainers import BiaffineNERTrainer
from .trainers import EntityRetrievalBiEncoderTrainer
from .trainers import EntityRerankingCrossEncoderTrainer
from .trainers import ATLOPTrainer

from . import evaluation

from . import utils


logger = logging.getLogger(__name__)


class Pipeline:

    def __init__(self, identifier, gpu_map=None):
        self.identifier = identifier
        if gpu_map is None:
            self.gpu_map = {
                "ner": 0,
                "ed_retrieval": 0,
                "ed_reranking": 0,
                "docre": 0,
            }
        else:
            self.gpu_map = gpu_map

        root_config = utils.get_hocon_config(os.path.join(
            expanduser("~"),
            ".kapipe",
            "config"
        ))

        if self.identifier is not None:
            self.pipe_config = root_config[self.identifier]
            self.ner = NER(
                task_config=self.pipe_config["ner"],
                gpu=self.gpu_map["ner"]
            )
            self.ed_ret = EDRetrieval(
                task_config=self.pipe_config["ed_retrieval"],
                gpu=self.gpu_map["ed_retrieval"]
            )
            self.ed_rank = EDReranking(
                task_config=self.pipe_config["ed_reranking"],
                gpu=self.gpu_map["ed_reranking"]
            )
            self.docre = DocRE(
                task_config=self.pipe_config["docre"],
                gpu=self.gpu_map["docre"]
            )
        else:
            self.pipe_config = None
            self.ner = NER(task_config=None, gpu=self.gpu_map["ner"])
            self.ed_ret = EDRetrieval(task_config=None, gpu=self.gpu_map["ed_retrieval"])
            self.ed_rank = EDReranking(task_config=None, gpu=self.gpu_map["ed_reranking"])
            self.docre = DocRE(task_config=None, gpu=self.gpu_map["docre"])
        
    def __call__(self, document, num_candidate_entities=10):
        """
        Parameters
        ----------
        document : Document
        num_candidate_entities: int
            by default 10

        Returns
        -------
        Document
        """
        document = self.ner(document=document)
        document, candidate_entities = self.ed_ret(
            document=document,
            num_candidate_entities=num_candidate_entities
        )
        document = self.ed_rank(
            document=document,
            candidate_entities=candidate_entities)
        document = self.docre(document=document)
        return document

    def save(self, identifier):
        self.pipe_config = {
            "ner": self.ner.task_config,
            "ed_retrieval": self.ed_ret.task_config,
            "ed_reranking": self.ed_rank.task_config,
            "docre": self.docre.task_config,
        }

        root_config = utils.get_hocon_config(os.path.join(
            expanduser("~"),
            ".kapipe",
            "config"
        ))

        root_config.update({
            identifier: self.pipe_config
        })

        utils.write_json(
            os.path.join(expanduser("~"), ".kapipe", "config"),
            root_config
        )


class NER:

    def __init__(self, task_config, gpu=0):
        self.task_config = task_config
        self.gpu = gpu
        self.device = f"cuda:{self.gpu}"

        if self.task_config is not None:
            self.system = BiaffineNERSystem(
                device=self.device,
                config=self.task_config["basepath"] + "/config",
                vocab_etype=self.task_config["basepath"] + "/entity_types.vocab.txt",
                path_model=self.task_config["basepath"] + "/model",
                verbose=False
            )

    def __call__(self, document):
        return self.system.extract(document=document)

    def fit(self, train_documents, dev_documents, optional_config=None):
        """  
        cf., https://github.com/norikinishida/kapipe/experiments/codes/run_biaffinener.py
        """
        prefix = utils.get_current_time()

        ##################
        # Set logger
        ##################

        base_output_path = os.path.join(
            expanduser("~"),
            ".kapipe",
            "results",
            "biaffinener",
            "default_config",
            prefix
        )
        utils.mkdir(base_output_path)

        root_logger = logging.getLogger()
        handler = logging.FileHandler(base_output_path + "/training.log", "w")
        root_logger.addHandler(handler)

        ##################
        # Get documents and vocabulary
        ##################
        
        entity_types = set()
        for documents in [train_documents, dev_documents]:
            for document in documents:
                for mention in document["mentions"]:
                    entity_types.add(mention["entity_type"])
        entity_types = sorted(list(entity_types))
        entity_types = ["NO-ENT"] + entity_types
        vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}

        ##################
        # Get system
        ##################

        trainer = BiaffineNERTrainer(base_output_path=base_output_path)
 
        root_config = utils.get_hocon_config(os.path.join(
            expanduser("~"),
            ".kapipe",
            "config"
        ))
        config = utils.get_hocon_config(root_config["default"]["ner"]["basepath"] + "/config")
        config["dataset_name"] = "NoName"
        config["allow_nested_entities"] = False
        if isinstance(optional_config, dict):
            config.update(optional_config)

        system = BiaffineNERSystem(
            device=self.device,
            config=config,
            vocab_etype=vocab_etype,
            path_model=None
        )

        ##################
        # Train
        ##################
        
        trainer.setup_dataset(
            system=system,
            documents=dev_documents,
            split="dev"
        )

        trainer.train(
            system=system,
            train_documents=train_documents,
            dev_documents=dev_documents
        )

        ##################
        # Finalize
        ##################
 
        self.task_config = {
            "basepath": base_output_path
        }

        self.system = BiaffineNERSystem(
            device=self.device,
            config=trainer.paths["path_config"],
            vocab_etype=trainer.paths["path_vocab_etype"],
            path_model=trainer.paths["path_snapshot"],
            verbose=False
        )

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 1
        handler = root_logger.handlers.pop()
        root_logger.removeHandler(handler)
        handler.close()
        logging.info(f"Removed {handler} from the root logger {root_logger}.")

 

 
class EDRetrieval:
    
    def __init__(self, task_config, gpu=0):
        self.task_config = task_config
        self.gpu = gpu
        self.device = f"cuda:{self.gpu}"

        if self.task_config is not None:
            self.system = EntityRetrievalBiEncoderSystem(
                device=self.device,
                config=self.task_config["basepath"] + "/config",
                path_entity_dict=self.task_config["entity_dict"],
                path_model=self.task_config["basepath"] + "/model",
                verbose=False
            )
            self.system.make_index(use_precomputed_entity_vectors=True)

    def __call__(self, document, num_candidate_entities=10):
        return self.system.extract(
            document=document,
            retrieval_size=num_candidate_entities
        )

    def fit(self, entity_dict, train_documents, dev_documents, optional_config=None):
        """  
        cf., https://github.com/norikinishida/kapipe/experiments/codes/run_entityretrievalbiencoder.py
        """
        prefix = utils.get_current_time()

        ##################
        # Set logger
        ##################

        base_output_path = os.path.join(
            expanduser("~"),
            ".kapipe",
            "results",
            "entityretrievalbiencoder",
            "default_config",
            prefix
        )
        utils.mkdir(base_output_path)

        root_logger = logging.getLogger()
        handler = logging.FileHandler(base_output_path + "/training.log", "w")
        root_logger.addHandler(handler)

        ##################
        # Get system
        ##################

        trainer = EntityRetrievalBiEncoderTrainer(
            base_output_path=base_output_path
        )

        root_config = utils.get_hocon_config(os.path.join(
            expanduser("~"),
            ".kapipe",
            "config"
        ))
        config = utils.get_hocon_config(root_config["default"]["ed_retrieval"]["basepath"] + "/config")
        config["dataset_name"] = "NoName"
        if isinstance(optional_config, dict):
            config.update(optional_config)

        # System requires a path to the entity dict
        # Thus, we save the entity_dict to the target path
        path_entity_dict = base_output_path + "/entity_dict.json"
        utils.write_json(path_entity_dict, entity_dict)
 
        system = EntityRetrievalBiEncoderSystem(
            device=self.device,
            config=config,
            path_entity_dict=path_entity_dict,
            path_model=None
        )

        ##################
        # Train
        ##################

        trainer.setup_dataset(
            system=system,
            documents=dev_documents,
            split="dev"
        )

        trainer.train(
            system=system,
            train_documents=train_documents,
            dev_documents=dev_documents
        )

        ##################
        # Finalize
        ##################

        self.task_config = {
            "basepath": base_output_path,
            "entity_dict": path_entity_dict
        }

        self.system = EntityRetrievalBiEncoderSystem(
            device=self.device,
            config=trainer.paths["path_config"],
            path_entity_dict=path_entity_dict,
            path_model=trainer.paths["path_snapshot"],
            verbose=False
        )
        self.system.make_index(use_precomputed_entity_vectors=True)
 
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 1
        handler = root_logger.handlers.pop()
        root_logger.removeHandler(handler)
        handler.close()
        logging.info(f"Removed {handler} from the root logger {root_logger}.")

 
class EDReranking:
    
    def __init__(self, task_config, gpu=0):
        self.task_config = task_config
        self.gpu = gpu
        self.device = f"cuda:{self.gpu}"

        if self.task_config is not None:
            self.system = EntityRerankingCrossEncoderSystem(
                device=self.device,
                config=self.task_config["basepath"] + "/config",
                path_entity_dict=self.task_config["entity_dict"],
                path_model=self.task_config["basepath"] + "/model",
                verbose=False
            )

    def __call__(self, document, candidate_entities):
        return self.system.extract(
            document=document,
            candidate_entities_for_doc=candidate_entities
        )

    def fit(
        self,
        entity_dict,
        train_documents,
        train_candidate_entities,
        dev_documents,
        dev_candidate_entities,
        optional_config=None
    ):
        """  
        cf., https://github.com/norikinishida/kapipe/experiments/codes/run_entityrerankingcrossencoder.py
        """
        prefix = utils.get_current_time()

        ##################
        # Set logger
        ##################

        base_output_path = os.path.join(
            expanduser("~"),
            ".kapipe",
            "results",
            "entityrerankingcrossencoder",
            "default_config",
            prefix
        )
        utils.mkdir(base_output_path)

        root_logger = logging.getLogger()
        handler = logging.FileHandler(base_output_path + "/training.log", "w")
        root_logger.addHandler(handler)

        ##################
        # Get documents
        ##################

        logger.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=train_candidate_entities,
                gold_path=train_documents
            )
        ))
        train_candidate_entities \
            = add_or_move_gold_entity_in_candidates(
                documents=train_documents,
                candidate_entities=train_candidate_entities
            )
        dev_candidate_entities \
            = add_or_move_gold_entity_in_candidates(
                documents=dev_documents,
                candidate_entities=dev_candidate_entities
            )
        logger.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=train_candidate_entities,
                gold_path=train_documents
            )
        ))
        logger.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=dev_candidate_entities,
                gold_path=dev_documents
            )
        ))

        ##################
        # Get system
        ##################

        trainer = EntityRerankingCrossEncoderTrainer(
            base_output_path=base_output_path
        )

        root_config = utils.get_hocon_config(os.path.join(
            expanduser("~"),
            ".kapipe",
            "config"
        ))
        config = utils.get_hocon_config(root_config["default"]["ed_reranking"]["basepath"] + "/config")
        config["dataset_name"] = "NoName"
        if isinstance(optional_config, dict):
            config.update(optional_config)

        # System requires a path to the entity dict
        # Thus, we save the entity_dict to the target path
        path_entity_dict = base_output_path + "/entity_dict.json"
        utils.write_json(path_entity_dict, entity_dict)
 
        system = EntityRerankingCrossEncoderSystem(
           device=self.device,
           config=config,
           path_entity_dict=path_entity_dict,
           path_model=None
        )
        
        ##################
        # Train
        ##################

        trainer.setup_dataset(
            system=system,
            documents=dev_documents,
            candidate_entities=dev_candidate_entities,
            split="dev"
        )
 
        trainer.train(
            system=system,
            train_documents=train_documents,
            train_candidate_entities=train_candidate_entities,
            dev_documents=dev_documents,
            dev_candidate_entities=dev_candidate_entities
        )

        ##################
        # Finalize
        ##################

        self.task_config = {
            "basepath": base_output_path,
            "entity_dict": path_entity_dict
        }

        self.system = EntityRerankingCrossEncoderSystem(
            device=self.device,
            config=trainer.paths["path_config"],
            path_entity_dict=path_entity_dict,
            path_model=trainer.paths["path_snapshot"],
            verbose=False
        )

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 1
        handler = root_logger.handlers.pop()
        root_logger.removeHandler(handler)
        handler.close()
        logging.info(f"Removed {handler} from the root logger {root_logger}.")

 
class DocRE:

    def __init__(self, task_config, gpu=0):
        self.task_config = task_config
        self.gpu = gpu
        self.device = f"cuda:{self.gpu}"

        if self.task_config is not None:
            self.system = ATLOPSystem(
                device=self.device,
                config=self.task_config["basepath"] + "/config",
                vocab_relation=self.task_config["basepath"] + "/relations.vocab.txt",
                path_model = self.task_config["basepath"] + "/model",
                verbose=False
            )

    def __call__(self, document):
        return self.system.extract(document=document)

    def fit(self, train_documents, dev_documents, optional_config=None):
        """  
        cf., https://github.com/norikinishida/kapipe/experiments/codes/run_atlop.py
        """
        prefix = utils.get_current_time()

        ##################
        # Set logger
        ##################

        base_output_path = os.path.join(
            expanduser("~"),
            ".kapipe",
            "results",
            "atlop",
            "default_config",
            prefix
        )
        utils.mkdir(base_output_path)

        root_logger = logging.getLogger()
        handler = logging.FileHandler(base_output_path + "/training.log", "w")
        root_logger.addHandler(handler)

        ##################
        # Get documents, vocabulary, and supplemental_info
        ##################

        relations = set()
        for documents in [train_documents, dev_documents]:
            for document in documents:
                for triple in document["relations"]:
                    relations.add(triple["relation"])
        relations = sorted(list(relations))
        relations = ["NO-REL"] + relations
        vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}

        supplemental_info = None
 
        ##################
        # Get system
        ##################

        trainer = ATLOPTrainer(base_output_path=base_output_path)

        root_config = utils.get_hocon_config(os.path.join(
            expanduser("~"),
            ".kapipe",
            "config"
        ))
        config = utils.get_hocon_config(root_config["default"]["docre"]["basepath"] + "/config")
        config["dataset_name"] = "NoName"
        config["possible_head_entity_types"] = None
        config["possible_tail_entity_types"] = None
        config["use_official_evaluation"] = False
        if isinstance(optional_config, dict):
            config.update(optional_config)

        system = ATLOPSystem(
            device=self.device,
            config=config,
            vocab_relation=vocab_relation,
            path_model=None
        )

        ##################
        # Train
        ##################

        trainer.setup_dataset(
            system=system,
            documents=train_documents,
            split="train",
            with_gold_annotations=True
        )
        trainer.setup_dataset(
            system=system,
            documents=dev_documents,
            split="dev",
            with_gold_annotations=True
        )

        trainer.train(
            system=system,
            train_documents=train_documents,
            dev_documents=dev_documents,
            supplemental_info=supplemental_info
        )

        ##################
        # Finalize
        ##################

        self.task_config = {
            "basepath": base_output_path
        }

        self.system = ATLOPSystem(
            device=self.device,
            config=trainer.paths["path_config"],
            vocab_relation=trainer.paths["path_vocab_relation"],
            path_model=trainer.paths["path_snapshot"],
            verbose=False
        )
 
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 1
        handler = root_logger.handlers.pop()
        root_logger.removeHandler(handler)
        handler.close()
        logging.info(f"Removed {handler} from the root logger {root_logger}.")

 
def add_or_move_gold_entity_in_candidates(
    documents,
    candidate_entities
):
    result_candidate_entities = []
    count_add = 0
    count_move = 0
    count_mentions = 0
    top_k = 0
    for document, cands_for_doc in zip(documents, candidate_entities):
        result_cands_for_mentions = []
        mentions = document["mentions"]
        cands_for_mentions = cands_for_doc["candidate_entities"]
        assert len(mentions) == len(cands_for_mentions)
        for mention, cands_for_mention in zip(mentions, cands_for_mentions):
            top_k = len(cands_for_mention)
            # Remove the gold entity in the candidates
            gold_entity_id = mention["entity_id"]
            cand_entity_ids = [c["entity_id"] for c in cands_for_mention]
            if gold_entity_id in cand_entity_ids:
                if gold_entity_id != cand_entity_ids[0]:
                    count_move += 1
                index = cand_entity_ids.index(gold_entity_id)
                gold_canonical_name = cands_for_mention[index]["canonical_name"]
                result_cands_for_mention \
                    = cands_for_mention[:index] + cands_for_mention[index+1:]
            else:
                count_add += 1
                gold_canonical_name = None
                result_cands_for_mention = cands_for_mention[:-1]
            count_mentions += 1
            # Append the gold entity to the top of the candidate list
            result_cands_for_mention = [{
                "entity_id": gold_entity_id,
                "canonical_name": gold_canonical_name,
                "score": 1000000.0
            }] + result_cands_for_mention
            result_cands_for_mentions.append(result_cands_for_mention)
        result_cands_for_doc = {
            "doc_key": document["doc_key"],
            "candidate_entities": result_cands_for_mentions
        }
        result_candidate_entities.append(result_cands_for_doc)
    logger.info(f"Added (or changed the position of) gold entities to the list of top-{top_k} candidate entities for {count_add} ({count_move}) / {count_mentions} mentions")
    return result_candidate_entities

