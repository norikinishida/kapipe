import copy
import logging
import os

# import numpy as np
# import torch
# from apex import amp
# import jsonlines
from tqdm import tqdm

# from . import shared_functions
from .. import evaluation
from .. import utils


logger = logging.getLogger(__name__)


class LLMEDTrainer:

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
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"

        # Paths to evaluation outputs and scores
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        return paths

    def setup_dataset(
        self,
        system,
        documents,
        candidate_entities,
        demonstrations,
        split
    ):
        """
        Parameters
        ----------
        system : LLMEDSystem
        documents : list[Document]
        candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        demonstrations : list[dict[str, list[Demo]]]
        split : str
        """
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(system.preprocessor.entity_dict.keys()))
            gold_documents = []
            for document, candidate_entities_for_doc in tqdm(
                zip(documents, candidate_entities),
                desc="dataset setup"
            ):
                gold_doc = copy.deepcopy(document)

                cands_for_mentions \
                    = candidate_entities_for_doc["candidate_entities"]
                mentions = document["mentions"]
                assert len(mentions) == len(cands_for_mentions)

                for m_i, (mention, cands_for_mention) in enumerate(zip(
                    mentions,
                    cands_for_mentions
                )):
                    cand_entity_ids = [
                        c["entity_id"] for c in cands_for_mention
                    ]
                    entity_id = mention["entity_id"]
                    in_kb = entity_id in kb_entity_ids
                    in_cand = entity_id in cand_entity_ids
                    gold_doc["mentions"][m_i]["in_kb"] = in_kb
                    gold_doc["mentions"][m_i]["in_cand"] = in_cand
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def save_system(self, system):
        """
        Parameters
        ----------
        system : ILCEDSystem
        """
        # Since we do not finetune the model, we save the configuration
        #   by this function.
        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], system.config)
        utils.write_json(self.paths["path_config"], system.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

    def evaluate(
        self,
        system,
        documents,
        candidate_entities,
        demonstrations,
        split,
        #
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        system : LLMEDSystem
        documents : list[Document]
        candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        demonstrations : list[dict[str, list[Demo]]]
        split : str
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # (documents, candidate_entities, demonstrations) -> path_pred
        result_documents = system.batch_extract(
            documents=documents,
            candidate_entities=candidate_entities,
            demonstrations=demonstrations
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        with open(
            self.paths[f"path_{split}_pred"].replace(".json", ".txt"), "w"
        ) as f:
            for result_doc in result_documents:
                doc_key = result_doc["doc_key"]
                prompt = result_doc["ed_prompt"]
                generated_text = result_doc["ed_generated_text"]
                f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                f.write(prompt + "\n\n")
                f.write(generated_text + "\n\n")
                f.flush()
        # (path_pred, path_gold) -> scores
        scores = evaluation.ed.accuracy(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        )
        scores.update(evaluation.ed.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        ))
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
