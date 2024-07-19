import copy
import logging
import os

# import numpy as np
# import torch
# from apex import amp
# import jsonlines
from tqdm import tqdm

# from .  import shared_functions
from .. import evaluation
from .. import utils


logger = logging.getLogger(__name__)


class LLMDocRETrainer:

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
        # Path to vocabulary
        paths["path_vocab_relation"] = self.base_output_path + "/relations.vocab.txt"

        # Path to gold training triples for Ign evaluation
        paths["path_gold_train_triples"] = self.base_output_path + "/gold_train_triples.json"

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
        demonstrations,
        split,
        with_gold_annotations=True
    ):
        """
        Parameters
        ----------
        system : LLMDocRESystem
        documents : list[Document]
        demonstrations : list[dict[str, list[DemoKeyInfo]]]
        split : str
        with_gold_annotations : bool
            by default True
        """
        # Cache the gold training triples for Ign evaluation
        if split == "train":
            if not os.path.exists(self.paths["path_gold_train_triples"]):
                gold_train_triples = []
                for document in tqdm(documents, desc="dataset setup"):
                    mentions = document["mentions"]
                    entity_index_to_mention_names = {
                        e_i: [
                            mentions[m_i]["name"]
                            for m_i in e["mention_indices"]
                        ]
                        for e_i, e in enumerate(document["entities"])
                    }
                    for triple in document["relations"]:
                        arg1_entity_i = triple["arg1"]
                        rel = triple["relation"]
                        arg2_entity_i = triple["arg2"]
                        arg1_mention_names \
                            = entity_index_to_mention_names[arg1_entity_i]
                        arg2_mention_names \
                            = entity_index_to_mention_names[arg2_entity_i]
                        for arg1_mention_name in arg1_mention_names:
                            for arg2_mention_name in arg2_mention_names:
                                gold_train_triples.append((
                                    arg1_mention_name,
                                    rel,
                                    arg2_mention_name
                                ))
                gold_train_triples = list(set(gold_train_triples))
                gold_train_triples = {"root": gold_train_triples}
                utils.write_json(
                    self.paths["path_gold_train_triples"],
                    gold_train_triples
                )
                logger.info(f"Saved the gold training triples for Ign evaluation in {self.paths['path_gold_train_triples']}")

        # Cache the gold annotations for evaluation
        if split != "train" and with_gold_annotations:
            path_gold = self.paths[f"path_{split}_gold"]
            if not os.path.exists(path_gold):
                gold_documents = []
                for document in tqdm(documents, desc="dataset setup"):
                    gold_doc = copy.deepcopy(document)
                    gold_documents.append(gold_doc)
                utils.write_json(path_gold, gold_documents)
                logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def save_system(self, system):
        """
        Parameters
        ----------
        system : LLMDocREsystem
        """
        # Since we do not finetune the model, we save the configuration
        #   and the relation type vocabulary by this function.
        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], system.config)
        utils.write_json(self.paths["path_config"], system.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])
        # Save the vocabulary (only once)
        utils.write_vocab(
            self.paths["path_vocab_relation"],
            system.vocab_relation,
            write_frequency=False
        )
        logger.info("Saved relation type vocabulary to %s" % self.paths["path_vocab_relation"])

    def evaluate(
        self,
        system,
        documents,
        demonstrations,
        split,
        supplemental_info,
        #
        skip_intra_inter=False,
        skip_ign=False,
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        system : LLMDocRESystem
        documents : list[Document]
        demonstrations : list[dict[str, list[Demo]]]
        split : str
        supplemental_info : dict[str, Any]
        skip_intra_inter : bool
            by default False
        skip_ign : bool
            by default False
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # (documents, demonstrations) -> path_pred
        result_documents = system.batch_extract(
            documents=documents,
            demonstrations=demonstrations
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        with open(
            self.paths[f"path_{split}_pred"].replace(".json", ".txt"), "w"
        ) as f:
            for result_doc in result_documents:
                doc_key = result_doc["doc_key"]
                prompt = result_doc["docre_prompt"]
                generated_text = result_doc["docre_generated_text"]
                f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                f.write(prompt + "\n\n")
                f.write(generated_text + "\n\n")
                f.write("------\n\n")
                f.flush()
        # (path_pred, path_gold) -> scores
        scores = evaluation.docre.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            skip_intra_inter=skip_intra_inter,
            skip_ign=skip_ign,
            gold_train_triples_path=self.paths["path_gold_train_triples"]
        )
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

    def official_evaluate(
        self,
        system,
        documents,
        demonstrations,
        split,
        supplemental_info,
        #
        prediction_only=False,
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        system : LLMDocRESystem
        documents : list[Document]
        demonstrations : list[dict[str, list[DemoKeyInfo]]]
        split : str
        supplemental_info : dict[str, Any]
        prediction_only : bool
            by default False
        get_scores_only : bool
            by default False

        Returns
        -------
        str[str, Any]
        """
        # NOTE: prediction_only is assumed to be used for the DocRED test set
        # (documents, demonstrations) -> path_pred
        result_documents = system.batch_extract(
            documents=documents,
            demonstrations=demonstrations
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        with open(
            self.paths[f"path_{split}_pred"].replace(".json", ".txt"), "w"
        ) as f:
            for result_doc in result_documents:
                doc_key = result_doc["doc_key"]
                prompt = result_doc["docre_prompt"]
                generated_text = result_doc["docre_generated_text"]
                f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                f.write(prompt + "\n\n")
                f.write(generated_text + "\n\n")
                f.write("------\n\n")
                f.flush()
        # path_pred -> triples (path_pred variant)
        triples = evaluation.docre.to_official(
            path_input=self.paths[f"path_{split}_pred"],
            path_output=
            self.paths[f"path_{split}_pred"].replace(".json", ".official.json")
        )
        if prediction_only:
            return
        # gold info
        original_data_dir = supplemental_info["original_data_dir"]
        train_file_name = supplemental_info["train_file_name"]
        dev_file_name = supplemental_info[f"{split}_file_name"]
        # (triples, gold info) -> scores
        scores = evaluation.docre.official_evaluate(
            triples=triples,
            original_data_dir=original_data_dir,
            train_file_name=train_file_name,
            dev_file_name=dev_file_name
        )
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
