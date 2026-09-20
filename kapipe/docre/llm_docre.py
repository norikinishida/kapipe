from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any

import torch
from tqdm import tqdm

from .. import evaluation
from .. import utils
from ..datatypes import (
    Document,
    Triple,
)
from ..llms import HuggingFaceLLM, OpenAILLM
from ..resources import resolve_snapshot_path
from .base import BaseDocRE


logger = logging.getLogger(__name__)


class LLMDocRE(BaseDocRE):
    """A class for performing document-level relation extraction using a large language model (LLM)."""

    @classmethod
    def from_identifier(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        identifier: str,
    ) -> "LLMDocRE":

        # Resolve the public identifier to the corresponding local snapshot
        snapshot_path = resolve_snapshot_path(
            component_name="docre",
            method_name="llm_docre",
            identifier=identifier,
        )

        # Load the extractor from the resolved snapshot
        extractor = cls.from_snapshot(
            model=model,
            snapshot_path=snapshot_path,
        )

        # Store the public identifier for later inspection
        extractor.identifier = identifier

        return extractor

    @classmethod
    def from_snapshot(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        snapshot_path: str,
    ) -> "LLMDocRE":

        # Define the default paths for the resources in the snapshot
        component_config_path = snapshot_path + "/component_config.json"
        vocab_path = snapshot_path + "/relations.vocab.txt"
        meta_info_path = snapshot_path + "/rel_meta_info.json"
        entity_dict_path = snapshot_path + "/entity_dict.json"
        demonstration_documents_path = (
            snapshot_path + "/demonstration_documents.json"
        )

        # Use the entity dictionary only when the snapshot provides it
        if not os.path.exists(entity_dict_path):
            entity_dict_path = None

        # Set the demonstration documents to None if the file does not exist
        if not os.path.exists(demonstration_documents_path):
            demonstration_documents_path = None

        # Load the component configuration
        component_config = utils.read_json(component_config_path)
        logger.info(f"Loaded component configuration from {component_config_path}")
        logger.info(utils.pretty_format_dict(component_config))

        # Initialize the extractor from explicit snapshot resources
        extractor = cls(
            # External
            model=model,
            # Internal
            **component_config,
            vocab_relation=vocab_path,
            rel_meta_info=meta_info_path,
            entity_dict_path=entity_dict_path,
            # Optional (Internal)
            demonstration_documents=demonstration_documents_path,
        )

        # Store the snapshot path for later inspection
        extractor.snapshot_path = snapshot_path

        return extractor

    def __init__(
        self,
        # External
        model: HuggingFaceLLM | OpenAILLM,
        # Internal
        knowledge_base_name: str,
        mention_style: str,
        with_span_annotation: bool,
        possible_head_entity_types: list[str] | None,
        possible_tail_entity_types: list[str] | None,
        vocab_relation: dict[str, int] | str,
        rel_meta_info: dict[str, dict[str, str]] | str,
        prompt_template_name_or_path: str = "docre_09_zeroshot",
        # Optional (Internal)
        entity_dict_path: str | None = None,
        demonstration_documents: list[Document] | str | None = None,
        # Optional
        **unused_kwargs: object,
    ):
        logger.info("########## LLMDocRE Initialization Starts ##########")

        self.model = model
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.knowledge_base_name = knowledge_base_name
        self.mention_style = mention_style
        self.with_span_annotation = with_span_annotation
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types

        # Load the relation vocabulary
        if isinstance(vocab_relation, str):
            vocab_path = vocab_relation
            vocab_relation = utils.read_vocab(vocab_path)
            logger.info(f"Loaded relation type vocabulary from {vocab_path}")
        self.vocab_relation = vocab_relation
        self.ivocab_relation = {
            relation_id: relation
            for relation, relation_id in self.vocab_relation.items()
        }

        # Load human-readable relation names and definitions. These values
        # are inserted into prompts and used to normalize generated labels.
        if isinstance(rel_meta_info, str):
            meta_path = rel_meta_info
            rel_meta_info = utils.read_json(meta_path)
            logger.info(f"Loaded relation meta-information from {meta_path}")
        self.rel_meta_info = rel_meta_info

        # Validate that the entity dictionary path is provided 
        # when using canonical entity names.
        if self.mention_style == "canonical_name" and entity_dict_path is None:
            raise ValueError(
                "entity_dict_path is required when mention_style is canonical_name"
            )

        # Load the entity dictionary only when it is provided
        if entity_dict_path is not None:
            logger.info(f"Loading entity dictionary from {entity_dict_path}")
            self.entity_dict = {
                epage["entity_id"]: epage
                for epage in utils.read_json(entity_dict_path)
            }
            logger.info(
                "Completed loading of entity dictionary with "
                f"{len(self.entity_dict)} entities "
                f"from {entity_dict_path}"
            )
        else:
            self.entity_dict = None

        # Load the demonstration documents
        if isinstance(demonstration_documents, str):
            demonstration_documents_path = demonstration_documents
            demonstration_documents = utils.read_json(demonstration_documents_path)
            logger.info(
                f"Loaded {len(demonstration_documents)} demonstration documents "
                f"from {demonstration_documents_path}"
            )
        elif demonstration_documents is None:
            # Use an empty list for zero-shot setting
            demonstration_documents = []
        self.demonstration_documents: list[Document] = demonstration_documents

        # Validate the mention style
        assert self.mention_style in [
            "all_mentions",
            "first_mention",
            "canonical_name",
        ]

        # Load the prompt template
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name="kapipe.docre.prompt_templates",
        )

        # Validate the prompt template
        if "{knowledge_base_name_prompt}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain "
                "{knowledge_base_name_prompt}."
            )
        if "{relations_prompt}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {relations_prompt}."
            )
        if "{test_case_prompt}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {test_case_prompt}."
            )

        # Set the prompt section for knowledge base name
        self.knowledge_base_name_prompt = self.knowledge_base_name

        # Generate the prompt section for relation labels
        self.relations_prompt = ""
        for relation in self.vocab_relation.keys():
            pretty_name = self.rel_meta_info[relation]["Pretty Name"]
            definition = self.rel_meta_info[relation]["Definition"]
            self.relations_prompt += f"- {pretty_name}: {definition}\n"
        self.relations_prompt = self.relations_prompt.rstrip()

        # Generate the prompt section for demonstrations
        self.demonstrations_prompt = self.generate_demonstrations_prompt()

        # Define regular expression for output parsing.
        # Parse lines of the following form:
        #
        #     - [Entity0] | [relation name] | [Entity1]
        #
        self.re_comp = re.compile(r"(.+?)\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$")

        # Create relation label mapping (normalized pretty name -> canonical name)
        # e.g., "chemical-induce-disease" -> "CID"
        self.normalized_to_canonical = {}
        for rel in self.vocab_relation.keys():
            pretty_name = self.rel_meta_info[rel]["Pretty Name"]
            normalized_pretty_name = pretty_name.lower()
            self.normalized_to_canonical[normalized_pretty_name] = rel

        logger.info("########## LLMDocRE Initialization Ends ##########")

    def save(self, snapshot_path: str) -> None:
        """Save the configuration, relation vocabulary, relation meta-information, entity dictionary, and demonstration pool to a specified snapshot path."""

        component_config_path = snapshot_path + "/component_config.json"
        vocab_path = snapshot_path + "/relations.vocab.txt"
        meta_info_path = snapshot_path + "/rel_meta_info.json"
        entity_dict_path = snapshot_path + "/entity_dict.json"
        demonstration_documents_path = (
            snapshot_path + "/demonstration_documents.json"
        )

        component_config: dict[str, Any] = {
            "prompt_template_name_or_path": self.prompt_template_name_or_path,
            "knowledge_base_name": self.knowledge_base_name,
            "mention_style": self.mention_style,
            "with_span_annotation": self.with_span_annotation,
            "possible_head_entity_types": self.possible_head_entity_types,
            "possible_tail_entity_types": self.possible_tail_entity_types,
        }

        utils.write_json(component_config_path, component_config)
        utils.write_vocab(vocab_path, self.vocab_relation, write_frequency=False)
        utils.write_json(meta_info_path, self.rel_meta_info)
        if self.entity_dict is not None:
            utils.write_json(entity_dict_path, list(self.entity_dict.values()))
        utils.write_json(demonstration_documents_path, self.demonstration_documents)

    def extract(
        self,
        document: Document,
    ) -> Document:
        """Extract triples from a single document."""

        # Skip relation extraction if there are 1 or fewer entities
        if len(document["entities"]) <= 1:
            result_document = copy.deepcopy(document)
            result_document["relations"] = []
            result_document["docre_prompt"] = ""
            result_document["docre_generated_text"] = ""
            return result_document

        with torch.no_grad():
            # Switch to inference mode for Hugging Face models
            if self.model.provider == "hf":
                self.model.llm.eval()

            # Generate the prompt
            prompt = self.generate_prompt(
                document=document,
            )
  
            # Generate a reponse
            generated_text = self.model.generate(prompt)

            # Parse the generated text into triples
            triples: list[Triple] = self.parse(
                document=document,
                generated_text=generated_text
            )

            # Integrate the triples into the document
            result_document = copy.deepcopy(document)
            result_document["relations"] = triples
            result_document["docre_prompt"] = prompt
            result_document["docre_generated_text"] = generated_text

            return result_document

    def generate_prompt(
        self,
        document: Document,
    ) -> str:
        """Generate the prompt for a given document and demonstration documents."""

        # Generate the prompt section for the test case
        test_case_prompt = self.generate_test_case_prompt(
            document=document,
        )

        # Combine all the prompt sections
        prompt = self.prompt_template.format(
            knowledge_base_name_prompt=self.knowledge_base_name_prompt,
            relations_prompt=self.relations_prompt,
            demonstrations_prompt=self.demonstrations_prompt,
            test_case_prompt=test_case_prompt
        )

        return prompt

    def generate_demonstrations_prompt(self) -> str:
        """Generate the prompt for the demonstration documents."""

        prompt = ""
        n_demos = len(self.demonstration_documents)

        for demo_i, demo_doc in enumerate(self.demonstration_documents):
            # Example ID
            prompt += f"Example {demo_i+1}:\n"

            # Input
            prompt += "Input Text:\n"
            prompt += f"{self.generate_input_text_prompt(document=demo_doc)}\n"
            prompt += "Entities:\n"
            prompt += f"{self.generate_input_entities_prompt(document=demo_doc)}\n"

            # Output
            prompt += "Output:\n"
            prompt += f"{self.generate_output_prompt(document=demo_doc)}\n"

            if demo_i < n_demos - 1:
                prompt += "\n"

        return prompt.rstrip()

    def generate_test_case_prompt(self, document: Document) -> str:
        """Generate the prompt for the test case."""

        # Input
        prompt = ""
        prompt += "Input Text:\n"
        prompt += f"{self.generate_input_text_prompt(document=document)}\n"
        prompt += "Entities:\n"
        prompt += f"{self.generate_input_entities_prompt(document=document)}\n"

        return prompt.rstrip()

    def generate_input_text_prompt(self, document: Document) -> str:
        """Generate the prompt for the input text."""

        prompt = " ".join(document["sentences"]) + "\n"

        return prompt.rstrip()

    def generate_input_entities_prompt(self, document: Document) -> str:
        """Generate the prompt for the input entities."""

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

    def generate_output_prompt(self, document: Document) -> str:
        """Generate the prompt for the output relations."""

        prompt = ""
        for triple in document["relations"]:
            head_idx = triple["arg1"]
            tail_idx = triple["arg2"]
            rel = triple["relation"]
            pretty_name = self.rel_meta_info[rel]["Pretty Name"]
            prompt += f"- Entity{head_idx} | {pretty_name} | Entity{tail_idx}\n"

        return prompt.rstrip()

    def parse(self, document: Document, generated_text: str) -> list[Triple]:
        """Parse the generated text into triples."""

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
                logger.info(
                    f"[{doc_key}] Skipped a generated line of invalid formatting: "
                    f"'{generated_line}'")
                continue
            _, head_id, relation, tail_id= parsed[0]

            # Check whether the head/tail IDs can be found in the possible list
            if (
                (head_id not in entity_id_to_index)
                or
                (tail_id not in entity_id_to_index)
                or
                head_id == tail_id
            ):
                logger.info(f"[{doc_key}] Skipped a generated line with invalid entity pair: '{generated_line}'")
                continue

            # Check whether the normalized relation label can be found in the possible set
            normalized_relation = relation.lower()
            if normalized_relation not in self.normalized_to_canonical:
                logger.info(
                    f"[{doc_key}] A generated line contains invalid relation: "
                    f"'{generated_line}'"
                )
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
            if tuple_ not in tuples:
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

    def submit_batch(
        self,
        documents: list[Document],
    ) -> list[str]:
        """Submit DocRE prompts and return the OpenAI Batch IDs.

        Pass the same documents in the same order to fetch_and_process_batch().
        Keep the model settings and prompt template unchanged between calls.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Generate prompts using the same method as extract().
        # Skip relation extraction if there are 1 or fewer entities.
        prompts: list[str] = []
        for document in documents:
            if len(document["entities"]) <= 1:
                continue
            prompt: str = self.generate_prompt(document=document)
            prompts.append(prompt)

        # Validate that there is at least one prompt to submit
        if len(prompts) == 0:
            raise ValueError(
                "No prompts to submit because all documents have 1 or fewer entities"
            )

        # Submit the prompts and get the Batch IDs
        batch_ids: list[str] = self.model.submit_batch(prompts=prompts)
        return batch_ids

    def fetch_and_process_batch(
        self,
        documents: list[Document],
        batch_ids: list[str],
    ) -> list[Document]:
        """Fetch responses and extract relations from the original documents.

        Require the same documents, order, model settings, and prompt template
        used at submission. Raise an error if the batch is not complete.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Fetch generated texts in the original request order
        generated_texts: list[str] = self.model.fetch_batch(batch_ids=batch_ids)

        # Validate that the number of generated texts matches the number of submitted documents
        submitted_documents: list[Document] = []
        for document in documents:
            # Match submit_batch(), which skips documents with 1 or fewer entities
            if len(document["entities"]) <= 1:
                continue
            submitted_documents.append(document)
        if len(generated_texts) != len(submitted_documents):
            raise ValueError(
                "The response count does not match the submitted document count"
            )

        # Process each Batch response using the same procedure as extract()
        result_documents: list[Document] = []
        generated_text_i = 0
        for document in documents:
            # Skip relation extraction if there are 1 or fewer entities
            if len(document["entities"]) <= 1:
                result_document = copy.deepcopy(document)
                result_document["relations"] = []
                result_document["docre_prompt"] = ""
                result_document["docre_generated_text"] = ""
                result_documents.append(result_document)
                continue

            # Regenerate the original prompt stored in the result
            prompt: str = self.generate_prompt(document=document)
            generated_text = generated_texts[generated_text_i]

            # Parse the generated text into triples
            triples: list[Triple] = self.parse(
                document=document,
                generated_text=generated_text
            )

            # Integrate the triples into the document
            result_document = copy.deepcopy(document)
            result_document["relations"] = triples
            result_document["docre_prompt"] = prompt
            result_document["docre_generated_text"] = generated_text
            result_documents.append(result_document)

            # Increment the index for the next generated text
            generated_text_i += 1

        return result_documents


#####################
# Trainer (Evaluator)
#####################


class LLMDocRETrainer:

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str, str]:
        paths = {}

        # configurations
        paths["snapshot_path"] = self.base_output_path
 
        # evaluation outputs
        paths["dev_gold_path"] = self.base_output_path + "/dev.gold.json"
        paths["dev_pred_path"] = self.base_output_path + "/dev.pred.json"
        paths["dev_eval_path"] = self.base_output_path + "/dev.eval.json"
        paths["test_gold_path"] = self.base_output_path + "/test.gold.json"
        paths["test_pred_path"] = self.base_output_path + "/test.pred.json"
        paths["test_eval_path"] = self.base_output_path + "/test.eval.json"

        # required for Ign evaluation
        paths["gold_train_triples_path"] = self.base_output_path + "/gold_train_triples.json"

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
            if not os.path.exists(self.paths["gold_train_triples_path"]):
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
                    self.paths["gold_train_triples_path"],
                    gold_train_triples
                )
                logger.info(f"Saved the gold training triples for Ign evaluation in {self.paths['gold_train_triples_path']}")

        # Cache the gold annotations for evaluation
        if split != "train" and with_gold_annotations:
            gold_path = self.paths[f"{split}_gold_path"]
            if not os.path.exists(gold_path):
                gold_documents = []
                for document in tqdm(documents, desc="dataset setup"):
                    gold_doc = copy.deepcopy(document)
                    gold_documents.append(gold_doc)
                utils.write_json(gold_path, gold_documents)
                logger.info(f"Saved the gold annotations for evaluation in {gold_path}")

    def save_extractor(self, extractor: LLMDocRE):
        extractor.save(snapshot_path=self.paths["snapshot_path"])

    def evaluate(
        self,
        extractor: LLMDocRE,
        documents: list[Document],
        split: str,
        supplemental_info: dict[str, Any],
        #
        skip_intra_inter: bool = False,
        skip_ign: bool = False,
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:

        # Apply the extractor
        result_documents: list[Document] = []
        for document in tqdm(
            documents,
            total=len(documents),
            desc="extraction steps"
        ):
            result_document = extractor.extract(document=document)
            result_documents.append(result_document)

        # Save the prediction results
        utils.write_json(self.paths[f"{split}_pred_path"], result_documents)

        # Save the prompt-response pairs in plain text
        with open(
            self.paths[f"{split}_pred_path"].replace(".json", ".txt"), "w"
        ) as f:
            for result_doc in result_documents:
                doc_key = result_doc["doc_key"]
                prompt = result_doc["docre_prompt"]
                generated_text = result_doc["docre_generated_text"]
                f.write("-------------------------------------\n\n")
                f.write(f"DOC_KEY: {doc_key}\n\n")
                f.write("PROMPT:\n")
                f.write(prompt + "\n\n")
                f.write("GENERATED TEXT:\n")
                f.write(generated_text + "\n\n")
                f.flush()

        if prediction_only:
            return

        # Calculate the evaluation scores
        scores = evaluation.docre.fscore(
            pred_path=self.paths[f"{split}_pred_path"],
            gold_path=self.paths[f"{split}_gold_path"],
            skip_intra_inter=skip_intra_inter,
            skip_ign=skip_ign,
            gold_train_triples_path=self.paths["gold_train_triples_path"]
        )

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"{split}_eval_path"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

    def official_evaluate(
        self,
        extractor: LLMDocRE,
        documents: list[Document],
        split: str,
        supplemental_info: dict[str, Any],
        #
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:

        # Apply the extractor
        result_documents: list[Document] = []
        for document in tqdm(
            documents,
            total=len(documents),
            desc="extraction steps"
        ):
            result_document = extractor.extract(document=document)
            result_documents.append(result_document)

        # Save the prediction results
        utils.write_json(self.paths[f"{split}_pred_path"], result_documents)

        with open(
            self.paths[f"{split}_pred_path"].replace(".json", ".txt"), "w"
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
            input_path=self.paths[f"{split}_pred_path"],
            output_path=
            self.paths[f"{split}_pred_path"].replace(".json", ".official.json")
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
        utils.write_json(self.paths[f"{split}_eval_path"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
