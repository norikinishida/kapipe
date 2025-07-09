from __future__ import annotations

import copy
# import json
import logging
import os
import re
from typing import Any

# import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm

from ..models import LLM
from ..models import OpenAILLM
from .. import evaluation
from .. import utils
from ..datatypes import (
    Config,
    Document,
    Triple,
    EntityPage,
    DemonstrationsForOneExample,
    ContextsForOneExample
)


logger = logging.getLogger(__name__)


class LLMDocRE:

    def __init__(
        self,
        device: str,
        # Initialization
        config: Config | str | None = None,
        vocab_relation: dict[str, int] | str | None = None,
        rel_meta_info: dict[str, dict[str, str]] | str | None = None,
        path_entity_dict: str | None = None,
        path_demonstration_pool: str | None = None,
        # Loading
        path_snapshot: str | None = None,
        # Misc.
        model: LLM | OpenAILLM | None = None
    ):
        logger.info("########## LLMDocRE Initialization Starts ##########")

        self.device = device
        self.path_snapshot = path_snapshot

        if path_snapshot is not None:
            assert config is None
            assert vocab_relation is None
            assert rel_meta_info is None
            assert path_entity_dict is None
            assert path_demonstration_pool is None
            config = path_snapshot + "/config"
            vocab_relation = path_snapshot + "/relations.vocab.txt"
            rel_meta_info = path_snapshot + "/rel_meta_info.json"
            path_entity_dict = path_snapshot + "/entity_dict.json"
            path_demonstration_pool = path_snapshot + "/demonstration_pool.json"
            if not os.path.exists(path_demonstration_pool):
                path_demonstration_pool = None

        # Load the configuration
        if isinstance(config, str):
            config_path = config
            config = utils.get_hocon_config(config_path=config_path)
            logger.info(f"Loaded configuration from {config_path}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

        # Load the relation vocabulary
        if isinstance(vocab_relation, str):
            vocab_path = vocab_relation
            vocab_relation = utils.read_vocab(vocab_path)
            logger.info(f"Loaded relation type vocabulary from {vocab_path}")
        self.vocab_relation = vocab_relation
        self.ivocab_relation = {i:l for l, i in self.vocab_relation.items()}

        # Load the relation meta information
        if isinstance(rel_meta_info, str):
            meta_path = rel_meta_info
            rel_meta_info = utils.read_json(meta_path)
            logger.info(f"Loaded relation meta-information from {meta_path}")
        self.rel_meta_info = rel_meta_info

        # Load the entity dictionary
        logger.info(f"Loading entity dictionary from {path_entity_dict}")
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        logger.info(f"Completed loading of entity dictionary with {len(self.entity_dict)} entities from {path_entity_dict}")

        # Initialize the prompt processor
        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            knowledge_base_name_prompt=config["knowledge_base_name"],
            vocab_relation=self.vocab_relation,
            rel_meta_info=self.rel_meta_info,
            entity_dict=self.entity_dict,
            mention_style=config["mention_style"],
            path_demonstration_pool=path_demonstration_pool,
            n_demonstrations=config["n_demonstrations"] ,
            with_span_annotation=config["with_span_annotation"]
        )

        # Initialize the model
        self.model_name = config["model_name"]
        assert self.model_name in ["hf", "openai"]
        if model is not None:
            self.model = model
            logger.info("LLM is provided by an argument")
        elif self.model_name == "hf":
            self.model = LLM(
                device=device,
                # Model
                llm_name_or_path=config["llm_name_or_path"],
                max_seg_len=config["max_seg_len"],
                quantization_bits=config["quantization_bits"],
                # Generation
                max_new_tokens=config["max_new_tokens"],
                beam_size=config["beam_size"],
                do_sample=config["do_sample"],
                num_return_sequences=config["num_return_sequences"],
                stop_list=config["stop_list"],
                clean_up_tokenization_spaces=config["clean_up_tokenization_spaces"],
            )
        else:
            self.model = OpenAILLM(
                openai_model_name=config["openai_model_name"],
                max_new_tokens=config["max_new_tokens"]
            )
        # self.model.llm.to(self.model.device)

        # Define regular expression for output parsing
        # self.re_comp = re.compile(
        #     "(.+?)\s*\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)$"
        # )
        # <bullet> <head entity ID> -> <relation> -> <tail entity ID>
        # self.re_comp = re.compile(
        #     "(.+?)\s*(.+?)\s*->\s*(.+?)\s*->\s*(.+?)$"
        # )
        # <bullet> <head entity ID> | <relation> | <tail entity ID>
        self.re_comp = re.compile("(.+?)\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$")

        # Create relation label mapping (normalized pretty name -> canonical name)
        # e.g., "chemical-induce-disease" -> "CID"
        self.normalized_to_canonical = {}
        for rel in self.vocab_relation.keys():
            pretty_name = self.rel_meta_info[rel]["Pretty Name"]
            self.normalized_to_canonical[pretty_name.lower()] = rel

        logger.info("########## LLMDocRE Initialization Ends ##########")

    def save(self, path_snapshot: str) -> None:
        path_config = path_snapshot + "/config"
        path_vocab = path_snapshot + "/relations.vocab.txt"
        path_meta_info = path_snapshot + "/rel_meta_info.json"
        path_entity_dict = path_snapshot + "/entity_dict.json"
        path_demonstration_pool = path_snapshot + "/demonstration_pool.json"
        utils.write_json(path_config, self.config)
        utils.write_vocab(path_vocab, self.vocab_relation, write_frequency=False)
        utils.write_json(path_meta_info, self.rel_meta_info)
        utils.write_json(path_entity_dict, list(self.entity_dict.values()))
        if self.prompt_processor.path_demonstration_pool is not None:
            utils.write_json(
                path_demonstration_pool,
                self.prompt_processor.demonstration_pool
            )

    def extract(
        self,
        document: Document,
        # optional: few-shot setting
        demonstrations_for_doc: DemonstrationsForOneExample | None = None,
        # optional: context augmentation
        contexts_for_doc: ContextsForOneExample | None = None
    ) -> Document:
        # Skip relation extraction if there are 1 or fewer entities
        if len(document["entities"]) <= 1:
            result_document = copy.deepcopy(document)
            result_document["relations"] = []
            result_document["docre_prompt"] = ""
            result_document["docre_generated_text"] = ""
            return result_document

        with torch.no_grad():
            if self.model_name == "hf":
                # Switch to inference mode
                self.model.llm.eval()

            # Generate a prompt
            prompt = self.prompt_processor.generate(
                document=document,
                demonstrations_for_doc=demonstrations_for_doc,
                contexts_for_doc=contexts_for_doc
            )
   
            if self.model_name == "hf": 
                # Preprocesss
                preprocessed_data = self.model.preprocess(prompt=prompt)

                # Tensorize
                model_input = self.model.tensorize(
                    preprocessed_data=preprocessed_data,
                    compute_loss=False
                )

                # Forward
                generated_text = self.model.generate(**model_input)[0]
                generated_text = self.model.remove_prompt_from_generated_text(
                    generated_text=generated_text
                )
            else:
                generated_text = self.model.generate(prompt)

            # Structurize
            triples: list[Triple] = self.structurize(
                document=document,
                generated_text=generated_text
            )

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["relations"] = triples
            if self.model_name == "hf":
                result_document["docre_prompt"] = preprocessed_data["prompt"]
            else:
                result_document["docre_prompt"] = prompt
            result_document["docre_generated_text"] = generated_text
            return result_document

    def structurize(self, document: Document, generated_text: str) -> list[Triple]:
        doc_key = document["doc_key"]

        # Get mapping from entity ID to entity index
        entity_id_to_index = {}
        for e_i, e in enumerate(document["entities"]):
            entity_id_to_index[f"Entity{e_i}"] = e_i
            
        tuples: list[tuple[int, str, int]] = []
        for generated_line in generated_text.split("\n"):
            generated_line = generated_line.strip()

            # Skip the empty line
            if generated_line == "":
                continue

            # Parse the generated line
            parsed = self.re_comp.findall(generated_line)
            if not (len(parsed) == 1 and len(parsed[0]) == 4):
                logger.info(f"[{doc_key}] Skipped a generated line of invalid formatting: '{generated_line}'")
                continue
            _, head_id, relation, tail_id= parsed[0]

            # Check whether the head/tail IDs can be found in the possible list
            if (
                (not head_id in entity_id_to_index)
                or
                (not tail_id in entity_id_to_index)
                or
                head_id == tail_id
            ):
                logger.info(f"[{doc_key}] Skipped a generated line with invalid entity pair: '{generated_line}'")
                continue

            # Check whether the normalized relation label can be found in the possible set
            normalized_relation = relation.lower()
            if not normalized_relation in self.normalized_to_canonical:
                logger.info(f"[{doc_key}] A generated line contains invalid relation: '{generated_line}'")
                # continue

            # Transform the normalized relation to canonical label
            canonical_relation = self.normalized_to_canonical.get(
                normalized_relation,
                relation
            )

            # Get entity index
            head_idx = entity_id_to_index[head_id]
            tail_idx = entity_id_to_index[tail_id]

            # Add a new tuple
            tuple_ = (head_idx, canonical_relation, tail_idx)
            if not tuple_ in tuples:
                tuples.append(tuple_)

        # Convert tuples
        triples: list[Triple] = []
        for (arg1, rel, arg2) in tuples:
            triples.append({
                "arg1": arg1,
                "relation": rel,
                "arg2": arg2
            })
        triples = sorted(
            triples,
            key=lambda x: (x["arg1"], x["arg2"], x["relation"])
        )
        return triples

    def batch_extract(
        self,
        documents: list[Document],
        # optional: few-shot setting
        demonstrations: list[DemonstrationsForOneExample] | None = None,
        # optional: context augmentation
        contexts: list[ContextsForOneExample] | None = None
    ) -> list[Document]:
        result_documents = []

        if demonstrations is None:
            demonstrations = [None] * len(documents)

        if contexts is None:
            contexts = [None] * len(documents)

        for document, demonstrations_for_doc, contexts_for_doc in tqdm(
            zip(documents, demonstrations, contexts),
            total=len(documents),
            desc="extraction steps"
        ):
            result_document = self.extract(
                document=document,
                demonstrations_for_doc=demonstrations_for_doc,
                contexts_for_doc=contexts_for_doc
            )
            result_documents.append(result_document)
        return result_documents


class PromptProcessor:

    def __init__(
        self,
        prompt_template_name_or_path: str,
        knowledge_base_name_prompt: str,
        vocab_relation: dict[str, int],
        rel_meta_info: dict[str, dict[str, str]],
        entity_dict: dict[str, EntityPage],
        mention_style: str,
        # optional: few-shot setting
        path_demonstration_pool: str | None = None,
        n_demonstrations: int | None = None,
        # misc.
        with_span_annotation: bool = True
    ):
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.knowledge_base_name_prompt = knowledge_base_name_prompt
        self.vocab_relation = vocab_relation
        self.rel_meta_info = rel_meta_info
        self.entity_dict = entity_dict
        self.mention_style = mention_style
        self.path_demonstration_pool = path_demonstration_pool
        self.n_demonstrations = n_demonstrations
        self.with_span_annotation = with_span_annotation

        assert self.mention_style in [
            "canonical_name", "first_mention", "all_mentions"
        ]

        # If demonstration pool is provided, `n_demonstration` should also be set
        if self.path_demonstration_pool is not None:
            assert self.n_demonstrations is not None
        if self.n_demonstrations == 0:
            self.path_demonstration_pool = None

        # Load the prompt template
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path
        )

        # Generate the prompt part for relation labels
        self.relations_prompt = ""
        for rel in vocab_relation.keys():
            pretty_name = self.rel_meta_info[rel]["Pretty Name"]
            definition = self.rel_meta_info[rel]["Definition"]
            self.relations_prompt += f"- {pretty_name}: {definition}\n"
        self.relations_prompt = self.relations_prompt.rstrip()

        # Load the demonstration pool
        if self.path_demonstration_pool is not None:
            self.demonstration_pool = {
                demo_doc["doc_key"]: demo_doc
                for demo_doc in utils.read_json(path_demonstration_pool)
            }

    def generate(
        self,
        document: Document,
        # optional: few-shot setting
        demonstrations_for_doc: DemonstrationsForOneExample | None = None,
        # optional: context augmentation
        contexts_for_doc: ContextsForOneExample | None = None
    ) -> str:
        ##########
        # Demonstrations Prompt
        ##########

        if demonstrations_for_doc is not None:
            # Create demonstration documents
            demonstration_documents: list[Document] = []
            for demo_key_dict in (
                demonstrations_for_doc["demonstrations"][:self.n_demonstrations]
            ):
                demo_doc = self.demonstration_pool[demo_key_dict["doc_key"]]
                demonstration_documents.append(demo_doc)
            # Generate prompt part for demonstrations
            demonstrations_prompt = self.generate_demonstrations_prompt(
                demonstration_documents=demonstration_documents
            )
        else:
            demonstrations_prompt = ""

        ##########
        # Contexts Prompt
        ##########

        if contexts_for_doc is not None:
            # Create contexts
            context_texts: list[str] = []
            for passage in contexts_for_doc["contexts"]:
                text = utils.create_text_from_passage(passage=passage, sep=" : ")
                context_texts.append(text)
            # Generate prompt part for contexts
            contexts_prompt = self.generate_contexts_prompt(
                context_texts=context_texts
            )
        else:
            contexts_prompt = ""

        ##########
        # Test Case Prompt
        ##########

        # Generate prompt part for test case
        test_case_prompt = self.generate_test_case_prompt(
            document=document,
        )

        ##########
        # Final Prompt
        ##########
 
        # Combine the prompt parts
        prompt = self.prompt_template.format(
            knowledge_base_name_prompt=self.knowledge_base_name_prompt,
            relations_prompt=self.relations_prompt,
            demonstrations_prompt=demonstrations_prompt,
            contexts_prompt=contexts_prompt,
            test_case_prompt=test_case_prompt
        )
        return prompt

    def generate_demonstrations_prompt(
        self,
        demonstration_documents: list[Document]
    ) -> str:
        prompt = ""
        n_demos = len(demonstration_documents)
        for demo_i, demo_doc in enumerate(demonstration_documents):
            prompt += f"Example {demo_i+1}:\n"
            prompt += f"Text: {self.generate_input_text_prompt(document=demo_doc)}\n"
            prompt += "Entities:\n"
            prompt += f"{self.generate_input_entities_prompt(document=demo_doc)}\n"
            prompt += "Output:\n"
            prompt += f"{self.generate_relations_prompt(document=demo_doc)}\n"
            if demo_i < n_demos - 1:
                prompt += "\n"
        return prompt.rstrip()

    def generate_contexts_prompt(self, context_texts: list[str]) -> str:
        n_contexts = len(context_texts)
        if n_contexts == 0:
            return ""
        else:
            prompt = ""
            for context_i, content in enumerate(context_texts):
                prompt += f"[{context_i+1}] {content.strip()} \n"
                if context_i < n_contexts - 1:
                    prompt += "\n"
            return prompt.rstrip()

    def generate_test_case_prompt(self, document: Document) -> str:
        prompt = ""
        prompt += f"Text: {self.generate_input_text_prompt(document=document)}\n"
        prompt += "Entities:\n"
        prompt += f"{self.generate_input_entities_prompt(document=document)}\n"
        return prompt.rstrip()

    def generate_input_text_prompt(self, document: Document) -> str:
        prompt = " ".join(document["sentences"]) + "\n"
        return prompt.rstrip()

    def generate_input_entities_prompt(self, document: Document) -> str:
        prompt = ""

        words = " ".join(document["sentences"]).split()

        mentions = document["mentions"]
        entities = document["entities"]

        for e_i, entity in enumerate(entities):
            entity_id = entity["entity_id"]
            entity_type = entity["entity_type"]

            if self.mention_style == "all_mentions":
                # Get mention names
                mention_indices = entity["mention_indices"]
                names = []
                for m_i in mention_indices:
                    # Get mention name
                    mention = mentions[m_i]
                    if not self.with_span_annotation:
                        name = mention["name"]
                    else:
                        begin_i, end_i = mention["span"]
                        name = " ".join(words[begin_i: end_i + 1])
                    # Remove duplicated mentions
                    # (inserted after the BioNLP'24 submission)
                    if name in names:
                        continue
                    names.append(name)
                # Add the entity to prompt
                names = ", ".join([f"\"{n}\"" for n in names])
                prompt += f"- Entity{e_i}: {names} ({entity_type})\n"
            elif self.mention_style == "first_mention":
                # Get the first mention name
                mention_indices = entity["mention_indices"]
                mention = mentions[mention_indices[0]]
                if self.with_span_annotation:
                    begin_i, end_i = mention["span"]
                    name = " ".join(words[begin_i: end_i + 1])
                else:
                    name = mention["name"]
                # Add the entity to prompt
                prompt += f"- Entity{e_i}: \"{name}\" ({entity_type})\n"
            elif self.mention_style == "canonical_name":
                # Get entity canonical name
                epage = self.entity_dict[entity_id]
                name = epage["canonical_name"]
                # Add the entity to prompt
                prompt += f"- Entity{e_i}: {name} ({entity_type})\n"
            else:
                raise Exception(f"Invalid mention_style: {self.mention_style}")
        return prompt.rstrip()

    def generate_relations_prompt(self, document: Document) -> str:
        prompt = ""
        for triple in document["relations"]:
            head_idx = triple["arg1"]
            tail_idx = triple["arg2"]
            rel = triple["relation"]
            pretty_name = self.rel_meta_info[rel]["Pretty Name"]
            prompt += f"- Entity{head_idx} | {pretty_name} | Entity{tail_idx}\n"
        return prompt.rstrip()
 

class LLMDocRETrainer:

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str, str]:
        paths = {}

        # configurations
        paths["path_snapshot"] = self.base_output_path
 
        # evaluation outputs
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        # required for Ign evaluation
        paths["path_gold_train_triples"] = self.base_output_path + "/gold_train_triples.json"

        return paths

    def setup_dataset(
        self,
        extractor: LLMDocRE,
        documents: list[Document],
        split: str,
        with_gold_annotations: bool = True
    ) -> None:
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

    def save_extractor(self, extractor: LLMDocRE):
        extractor.save(path_snapshot=self.paths["path_snapshot"])

    def evaluate(
        self,
        extractor: LLMDocRE,
        documents: list[Document],
        demonstrations: list[DemonstrationsForOneExample],
        contexts: list[ContextsForOneExample],
        split: str,
        supplemental_info: dict[str, Any],
        #
        skip_intra_inter: bool = False,
        skip_ign: bool = False,
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:
        # Apply the extractor
        result_documents = extractor.batch_extract(
            documents=documents,
            demonstrations=demonstrations,
            contexts=contexts
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

        if prediction_only:
            return

        # Calculate the evaluation scores
        scores = evaluation.docre.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            skip_intra_inter=skip_intra_inter,
            skip_ign=skip_ign,
            gold_train_triples_path=self.paths["path_gold_train_triples"]
        )

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

    def official_evaluate(
        self,
        extractor: LLMDocRE,
        documents: list[Document],
        demonstrations: list[DemonstrationsForOneExample],
        contexts: list[ContextsForOneExample],
        split: str,
        supplemental_info: dict[str, Any],
        #
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:
        # Apply the extractor
        result_documents = extractor.batch_extract(
            documents=documents,
            demonstrations=demonstrations,
            contexts=contexts
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

        triples = evaluation.docre.to_official(
            path_input=self.paths[f"path_{split}_pred"],
            path_output=
            self.paths[f"path_{split}_pred"].replace(".json", ".official.json")
        )

        if prediction_only:
            return

        # Calculate the evaluation scores
        original_data_dir = supplemental_info["original_data_dir"]
        train_file_name = supplemental_info["train_file_name"]
        dev_file_name = supplemental_info[f"{split}_file_name"]
        scores = evaluation.docre.official_evaluate(
            triples=triples,
            original_data_dir=original_data_dir,
            train_file_name=train_file_name,
            dev_file_name=dev_file_name
        )

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
