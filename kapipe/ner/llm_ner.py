from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any
import unicodedata

import torch
from tqdm import tqdm

from .. import evaluation
from .. import utils
from ..datatypes import (
    Document,
    Mention,
)
from ..llms import HuggingFaceLLM, OpenAILLM
from ..resources import resolve_snapshot_path
from .base import BaseNER


logger = logging.getLogger(__name__)


class LLMNER(BaseNER):

    @classmethod
    def from_identifier(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        identifier: str,
    ) -> "LLMNER":

        # Resolve the public identifier to the corresponding local snapshot
        snapshot_path = resolve_snapshot_path(
            component_name="ner",
            method_name="llm_ner",
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
    ) -> "LLMNER":

        # Define the default paths for the resources in the snapshot
        component_config_path = snapshot_path + "/component_config.json"
        vocab_path = snapshot_path + "/entity_types.vocab.txt"
        meta_info_path = snapshot_path + "/etype_meta_info.json"
        demonstration_documents_path = (
            snapshot_path + "/demonstration_documents.json"
        )

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
            vocab_etype=vocab_path,
            etype_meta_info=meta_info_path,
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
        vocab_etype: dict[str, int] | str,
        etype_meta_info: dict[str, dict[str, str]] | str,
        prompt_template_name_or_path: str = "ner_14_zeroshot",
        # Optional (Internal)
        demonstration_documents: list[Document] | str | None = None,
        # Optional
        **unused_kwargs: object,
    ):
        logger.info("########## LLMNER Initialization Starts ##########")

        self.model = model
        self.prompt_template_name_or_path = prompt_template_name_or_path

        # Load the entity-type vocabulary
        if isinstance(vocab_etype, str):
            vocab_path = vocab_etype
            vocab_etype = utils.read_vocab(vocab_path)
            logger.info(f"Loaded entity type vocabulary from {vocab_path}")
        self.vocab_etype = vocab_etype
        self.ivocab_etype = {
            entity_type_id: entity_type
            for entity_type, entity_type_id in self.vocab_etype.items()
        }

        # Load human-readable entity-type names and definitions. These values
        # are inserted into prompts and used to normalize generated labels.
        if isinstance(etype_meta_info, str):
            meta_path = etype_meta_info
            etype_meta_info = utils.read_json(meta_path)
            logger.info(f"Loaded entity type meta-information from {meta_path}")
        self.etype_meta_info = etype_meta_info

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

        # Load the prompt template
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name="kapipe.ner.prompt_templates",
        )

        # Validate the prompt template
        if "{entity_types_prompt}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {entity_types_prompt}."
            )
        if "{test_case_prompt}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {test_case_prompt}."
            )

        # Generate the prompt section for entity types
        self.entity_types_prompt = ""
        for etype in vocab_etype.keys():
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            definition = self.etype_meta_info[etype]["Definition"]
            self.entity_types_prompt += f"- {pretty_name}: {definition}\n"
        self.entity_types_prompt = self.entity_types_prompt.rstrip()

        # Generate the prompt section for demonstrations
        self.demonstrations_prompt = self.generate_demonstrations_prompt()

        # Define regular expression for output parsing.
        # Parse generated lines of the followingform:
        #
        #     - [mention text] | [entity type]
        #
        self.re_comp = re.compile(r"(.+?)\s*(.+?)\s*\|\s*(.+?)$")

        # Create entity type mapping (normalized pretty name -> canonical name)
        # e.g., "Location" -> "LOC"
        self.normalized_to_canonical: dict[str, str] = {}
        for etype in self.vocab_etype.keys():
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            normalized_pretty_name = pretty_name.lower()
            self.normalized_to_canonical[normalized_pretty_name] = etype

        logger.info("########## LLMNER Initialization Ends ##########")

    def save(self, snapshot_path: str) -> None:
        """Save the configuration, entity-type vocabulary, meta-information, and demonstration documents to a snapshot."""

        component_config_path = snapshot_path + "/component_config.json"
        vocab_path = snapshot_path + "/entity_types.vocab.txt"
        meta_info_path = snapshot_path + "/etype_meta_info.json"
        demonstration_documents_path = snapshot_path + "/demonstration_documents.json"

        component_config: dict[str, Any] = {
            "prompt_template_name_or_path": self.prompt_template_name_or_path,
        }

        utils.write_json(component_config_path, component_config)
        utils.write_vocab(vocab_path, self.vocab_etype, write_frequency=False)
        utils.write_json(meta_info_path, self.etype_meta_info)
        utils.write_json(demonstration_documents_path, self.demonstration_documents)

    def extract(
        self,
        document: Document,
    ) -> Document:
        """Extract named entity mentions from a single document."""

        with torch.no_grad():
            # Switch to inference mode for Hugging Face models
            if self.model.provider == "hf":
                self.model.llm.eval()

            # Generate the prompt
            prompt = self.generate_prompt(
                document=document,
            )

            # Generate the response
            generated_text = self.model.generate(prompt)

            # Parse the generated text into mentions
            mentions = self.parse(
                document=document,
                generated_text=generated_text
            )

            # Integrate the mentions into the document
            result_document = copy.deepcopy(document)
            result_document["mentions"] = mentions
            result_document["ner_prompt"] = prompt
            result_document["ner_generated_text"] = generated_text

            return result_document

    def generate_prompt(
        self,
        document: Document,
    ) -> str:
        """Generate the prompt for the input document."""

        # Generate the prompt section for the test case
        test_case_prompt = self.generate_test_case_prompt(
            document=document
        )

        # Combine all the prompt sections
        prompt = self.prompt_template.format(
            entity_types_prompt=self.entity_types_prompt,
            demonstrations_prompt=self.demonstrations_prompt,
            test_case_prompt=test_case_prompt
        )

        return prompt

    def generate_demonstrations_prompt(self) -> str:
        """Generate the prompt for the demonstrations."""

        prompt = ""
        n_demos = len(self.demonstration_documents)
        for demo_i, demo_doc in enumerate(self.demonstration_documents):
            # Example ID
            prompt += f"Example {demo_i+1}:\n"

            # Input
            prompt += "Input:\n"
            prompt += f"{self.generate_input_text_prompt(document=demo_doc)}\n"

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
        prompt += "Input:\n"
        prompt += f"{self.generate_input_text_prompt(document=document)}\n"

        return prompt.rstrip()

    def generate_input_text_prompt(self, document: Document) -> str:
        """Generate the prompt for the input text."""

        prompt = " ".join(document["sentences"]) + "\n"

        return prompt.rstrip()

    def generate_output_prompt(self, document: Document) -> str:
        """Generate the prompt for the output mentions."""

        prompt = ""
        words = " ".join(document["sentences"]).split()
        for mention in document["mentions"]:
            begin_i, end_i = mention["span"]
            name = " ".join(words[begin_i: end_i + 1])
            etype = mention["entity_type"]
            if etype in self.etype_meta_info:
                pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            else:
                pretty_name = etype
            prompt += f"- {name} | {pretty_name}\n"

        return prompt.rstrip()

    def parse(self, document: Document, generated_text: str) -> list[Mention]:
        """Parse the generated text into the mentions."""

        doc_key = document["doc_key"]

        # Get mapping from character position to word position (index)
        original_words = " ".join(document["sentences"]).split()
        normalized_words = [
            unicodedata.normalize("NFC", w).lower() for w in original_words
        ]
        normalized_text = " ".join(normalized_words)
       
        # NOTE: `char_index_to_word_index` must be created based on
        #       the normalized (lowered) text.
        char_index_to_word_index: list[int] = []
        for w_i, w in enumerate(normalized_words):
            # n_chars = len(w)
            # char_index_to_word_index.extend([w_i] * n_chars)
            # char_index_to_word_index.append(None) # for space between words
            if w_i > 0:
                char_index_to_word_index.append(None) # space
            for _ in w:
                char_index_to_word_index.append(w_i)

        # Create a mapping from token index to sentence index
        token_index_to_sent_index: list[int] = []
        for s_i, sent in enumerate(document["sentences"]):
            s_len = len(sent.split())
            token_index_to_sent_index.extend([s_i] * s_len)

        # Parse the generated text and extract mention tuples
        # (begin_token_index, end_token_index, entity_type)
        tuples: list[tuple[int, int, str]] = []
        for generated_line in generated_text.split("\n"):
            generated_line = generated_line.strip()

            # Skip the empty line
            if generated_line == "":
                continue

            # Parse the generated line
            parsed = self.re_comp.findall(generated_line)
            if not (len(parsed) == 1 and len(parsed[0]) == 3):
                logger.info(f"[{doc_key}] Skipped a generated line of invalid formatting: '{generated_line}'")
                continue
            _, name, entity_type = parsed[0]

            # Check whether the mention can be found in the input text
            # i.e., get word-level spans
            normalized_name = name.lower()
            normalized_name = unicodedata.normalize("NFC", normalized_name)
            spans = self.extract_word_level_spans(
                normalized_name=normalized_name,
                normalized_text=normalized_text,
                char_index_to_word_index=char_index_to_word_index
            )

            # Remove cross-sentence spans
            spans = [
                (b,e) for b,e in spans
                if token_index_to_sent_index[b] == token_index_to_sent_index[e]
            ]

            # Remove very long spans
            spans = [(b,e) for b,e in spans if (e - b) <= 10]

            # Skip this line if no mention string is detected
            if len(spans) == 0:
                logger.info(f"[{doc_key}] Skipped a generated line with invalid mention: '{generated_line}'")
                continue

            # Check whether the entity type can be found in the possible list
            normalized_entity_type = entity_type.lower()
            if not normalized_entity_type in self.normalized_to_canonical:
                logger.info(f"[{doc_key}] A generated line contains invalid entity type: '{generated_line}'")
                # continue

            # Map the normalized entity type to the canonical entity type
            canonical_entity_type = self.normalized_to_canonical.get(
                normalized_entity_type,
                entity_type
            )

            # Add new tuples
            for begin_token_i, end_token_i in spans:
                tuple_ = (begin_token_i, end_token_i, canonical_entity_type)
                if not tuple_ in tuples:
                    tuples.append(tuple_)

        # Convert the tuples
        mentions: list[Mention] = []
        for (begin_i, end_i, etype) in tuples:
            name = " ".join(original_words[begin_i: end_i + 1])
            mentions.append({
                "span": (begin_i, end_i),
                "name": name,
                "entity_type": etype,
            })
        mentions = sorted(mentions, key=lambda m: m["span"])

        return mentions

    def extract_word_level_spans(
        self,
        normalized_name: str,
        normalized_text: str,
        char_index_to_word_index: list[int]
    ) -> list[tuple[int, int]]:
        """Extract word-level spans of a mention string in the input text."""

        spans: list[tuple[int, int]] = []
        pattern = r"\s*".join(re.escape(c) for c in normalized_name)
        results = re.finditer(
            " " + pattern + " ",
            " " + normalized_text + " "
        )
        for result in results:
            begin_char_i, end_char_i = result.span()
            begin_char_i += 1 # remove leading space
            end_char_i -= 1 # remove trailing space
            begin_char_i -= 1 # remove initial space added to normalized_text
            end_char_i -= 1
            begin_word_i = char_index_to_word_index[begin_char_i]
            end_word_i = char_index_to_word_index[end_char_i - 1]
            spans.append((begin_word_i, end_word_i))

        return spans

    def submit_batch(
        self,
        documents: list[Document],
    ) -> list[str]:
        """Submit NER prompts and return the OpenAI Batch IDs.

        Pass the same documents in the same order to fetch_and_process_batch().
        Keep the model settings and prompt template unchanged between calls.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Generate prompts using the same method as extract()
        prompts: list[str] = []
        for document in documents:
            prompt: str = self.generate_prompt(document=document)
            prompts.append(prompt)

        # Submit the prompts and get the Batch IDs
        batch_ids: list[str] = self.model.submit_batch(prompts=prompts)
        return batch_ids

    def fetch_and_process_batch(
        self,
        documents: list[Document],
        batch_ids: list[str],
    ) -> list[Document]:
        """Fetch responses and extract mentions from the original documents.

        Require the same documents, order, model settings, and prompt template
        used at submission. Raise an error if the batch is not complete.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Fetch generated texts in the original request order
        generated_texts: list[str] = self.model.fetch_batch(batch_ids=batch_ids)

        # Validate that the number of generated texts matches the number of documents
        if len(generated_texts) != len(documents):
            raise ValueError("The response count does not match the document count")

        # Process each Batch response using the same procedure as extract()
        result_documents: list[Document] = []
        for document, generated_text in zip(
            documents,
            generated_texts,
            strict=True,
        ):
            # Regenerate the original prompt stored in the result
            prompt: str = self.generate_prompt(document=document)

            # Parse the generated text into mentions
            mentions = self.parse(
                document=document,
                generated_text=generated_text
            )

            # Integrate the mentions into the document
            result_document = copy.deepcopy(document)
            result_document["mentions"] = mentions
            result_document["ner_prompt"] = prompt
            result_document["ner_generated_text"] = generated_text
            result_documents.append(result_document)

        return result_documents


#####################
# Trainer (Evaluator)
#####################


class LLMNERTrainer:

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

        return paths

    def setup_dataset(
        self,
        extractor: LLMNER,
        documents: list[Document],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        gold_path = self.paths[f"{split}_gold_path"]
        if not os.path.exists(gold_path):
            gold_documents = []
            for document in tqdm(documents, desc="dataset setup"):
                gold_doc = copy.deepcopy(document)
                gold_documents.append(gold_doc)
            utils.write_json(gold_path, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {gold_path}")

    def save_extractor(self, extractor: LLMNER) -> None:
        extractor.save(snapshot_path=self.paths["snapshot_path"])

    def evaluate(
        self,
        extractor: LLMNER,
        documents: list[Document],
        split: str,
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
            result_document = extractor.extract(
                document=document,
            )
            result_documents.append(result_document)

        # Save the prediction results
        utils.write_json(self.paths[f"{split}_pred_path"], result_documents)

        # Save the prompt-response pairs in plain text
        with open(
            self.paths[f"{split}_pred_path"].replace(".json", ".txt"), "w"
        ) as f:
            for result_doc in result_documents:
                doc_key = result_doc["doc_key"]
                prompt = result_doc["ner_prompt"]
                generated_text = result_doc["ner_generated_text"]
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
        scores = evaluation.ner.fscore(
            pred_path=self.paths[f"{split}_pred_path"],
            gold_path=self.paths[f"{split}_gold_path"]
        )

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"{split}_eval_path"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
