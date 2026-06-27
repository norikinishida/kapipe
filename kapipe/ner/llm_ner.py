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
    Config,
    Document,
    Mention,
    ContextsForOneExample
)
from ..llms import HuggingFaceLLM, OpenAILLM
from ..resources import resolve_snapshot_path


logger = logging.getLogger(__name__)


class LLMNER:

    @classmethod
    def from_identifier(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        identifier: str,
    ) -> "LLMNER":

        # Resolve the public identifier to the corresponding local snapshot
        path_snapshot = resolve_snapshot_path(
            component_name="ner",
            method_name="llm_ner",
            identifier=identifier,
        )

        # Load the extractor from the resolved snapshot
        extractor = cls.from_snapshot(
            model=model,
            path_snapshot=path_snapshot,
        )

        # Store the public identifier for later inspection
        extractor.identifier = identifier

        return extractor

    @classmethod
    def from_snapshot(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        path_snapshot: str,
    ) -> "LLMNER":

        # Define the default paths for the resources in the snapshot
        path_config = path_snapshot + "/config"
        path_vocab = path_snapshot + "/entity_types.vocab.txt"
        path_meta_info = path_snapshot + "/etype_meta_info.json"
        path_demonstration_documents = (
            path_snapshot + "/demonstration_documents.json"
        )

        # Use an empty list of demonstrations if the snapshot does not contain 
        # any demonstration documents.
        if os.path.exists(path_demonstration_documents):
            demonstration_documents = path_demonstration_documents
        else:
            demonstration_documents = []

        # Initialize the extractor from explicit snapshot resources
        extractor = cls(
            model=model,
            config=path_config,
            vocab_etype=path_vocab,
            etype_meta_info=path_meta_info,
            demonstration_documents=demonstration_documents,
        )

        # Store the snapshot path for later inspection
        extractor.path_snapshot = path_snapshot

        return extractor

    def __init__(
        self,
        model: HuggingFaceLLM | OpenAILLM,
        config: Config | str | None = None,
        vocab_etype: dict[str, int] | str | None = None,
        etype_meta_info: dict[str, dict[str, str]] | str | None = None,
        demonstration_documents: list[Document] | str | None = None,
    ):
        logger.info("########## LLMNER Initialization Starts ##########")

        self.model = model

        # Load the configuration
        if isinstance(config, str):
            config_path = config
            config = utils.get_hocon_config(config_path=config_path)
            logger.info(f"Loaded configuration from {config_path}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

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
            path_demonstration_documents = demonstration_documents
            demonstration_documents = utils.read_json(path_demonstration_documents)
            logger.info(
                f"Loaded {len(demonstration_documents)} demonstration documents "
                f"from {path_demonstration_documents}"
            )
        elif demonstration_documents is None:
            # Use an empty list for zero-shot setting
            demonstration_documents = []
        self.demonstration_documents: list[Document] = demonstration_documents

        # Initialize the prompt processor, which generates prompts for the LLM
        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=(
                self.config["prompt_template_name_or_path"]
            ),
            vocab_etype=self.vocab_etype,
            etype_meta_info=self.etype_meta_info,
        )

        # Check the LLM provider
        self.provider = self.config["provider"]
        if self.provider not in ["hf", "openai"]:
            raise ValueError(f"Invalid provider: {self.provider}")
        logger.info("LLM is provided by an argument")

        # Define regular expression for output parsing.
        # Parse generated lines of the followingform:
        #
        #     - [mention text] | [entity type]
        #
        self.re_comp = re.compile("(.+?)\s*(.+?)\s*\|\s*(.+?)$")

        # Create entity type mapping (normalized pretty name -> canonical name)
        # e.g., "Location" -> "LOC"
        self.normalized_to_canonical: dict[str, str] = {}
        for etype in self.vocab_etype.keys():
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            normalized_pretty_name = pretty_name.lower()
            self.normalized_to_canonical[normalized_pretty_name] = etype

        logger.info("########## LLMNER Initialization Ends ##########")

    def save(self, path_snapshot: str) -> None:
        """Save the configuration, entity-type vocabulary, meta-information, and demonstration documents to a snapshot."""

        path_config = path_snapshot + "/config"
        path_vocab = path_snapshot + "/entity_types.vocab.txt"
        path_meta_info = path_snapshot + "/etype_meta_info.json"
        path_demonstration_documents = path_snapshot + "/demonstration_documents.json"

        utils.write_json(path_config, self.config)
        utils.write_vocab(path_vocab, self.vocab_etype, write_frequency=False)
        utils.write_json(path_meta_info, self.etype_meta_info)
        utils.write_json(path_demonstration_documents, self.demonstration_documents)

    def extract(
        self,
        document: Document,
        # Optional: context augmentation
        contexts_for_doc: ContextsForOneExample | None = None
    ) -> Document:
        """Extract named entity mentions from a single document."""

        with torch.no_grad():
            # Switch to inference mode for Hugging Face models
            if self.provider == "hf":
                self.model.llm.eval()

            # Generate the prompt
            prompt = self.prompt_processor.generate(
                document=document,
                demonstration_documents=self.demonstration_documents,
                contexts_for_doc=contexts_for_doc,
            )

            # Generate the response
            generated_text = self.model.generate(prompt)

            # Structurize the generated text into mentions
            mentions = self.structurize(
                document=document,
                generated_text=generated_text
            )

            # Integrate the mentions into the document
            result_document = copy.deepcopy(document)
            result_document["mentions"] = mentions
            result_document["ner_prompt"] = prompt
            result_document["ner_generated_text"] = generated_text

            return result_document

    def structurize(self, document: Document, generated_text: str) -> list[Mention]:
        """Structurize the generated text into the mentions."""

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

    def batch_extract(
        self,
        documents: list[Document],
        # optional: context augmentation
        contexts: list[ContextsForOneExample] | None = None
    ) -> list[Document]:
        """Extract named entity mentions from a batch of documents."""

        result_documents: list[Document] = []

        # Use empty contexts when no contexts are provided
        if contexts is None:
            contexts = [None] * len(documents)

        for document, contexts_for_doc in tqdm(
            zip(documents, contexts),
            total=len(documents),
            desc="extraction steps"
        ):
            result_document = self.extract(
                document=document,
                contexts_for_doc=contexts_for_doc
            )
            result_documents.append(result_document)

        return result_documents


class PromptProcessor:

    def __init__(
        self,
        prompt_template_name_or_path: str,
        vocab_etype: dict[str, int],
        etype_meta_info: dict[str, dict[str, str]],
    ):
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.vocab_etype = vocab_etype
        self.etype_meta_info = etype_meta_info

        # Load the prompt template
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name="kapipe.ner.prompt_templates",
        )

        # Generate the prompt part for entity types
        self.entity_types_prompt = ""
        for etype in vocab_etype.keys():
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            definition = self.etype_meta_info[etype]["Definition"]
            self.entity_types_prompt += f"- {pretty_name}: {definition}\n"
        self.entity_types_prompt = self.entity_types_prompt.rstrip()

    def generate(
        self,
        document: Document,
        demonstration_documents: list[Document],
        # Optional: context augmentation
        contexts_for_doc: ContextsForOneExample | None = None
    ) -> str:
        """Generate a prompt for the input document."""

        ##########
        # Demonstrations Prompt
        ##########

        # Generate the prompt part for the demonstrations
        demonstrations_prompt = self.generate_demonstrations_prompt(
            demonstration_documents=demonstration_documents,
        )        

        ##########
        # Contexts Prompt
        ##########

        if contexts_for_doc is not None:
            # Create contexts
            context_texts: list[str] = []
            for passage in contexts_for_doc["contexts"]:
                text = utils.create_text_from_passage(passage=passage, sep=" : ")
                context_texts.append(text)
            # Generate the prompt part for the contexts
            contexts_prompt = self.generate_contexts_prompt(
                context_texts=context_texts
            )
        else:
            contexts_prompt = ""

        ##########
        # Test Case Prompt
        ##########

        # Generate the prompt part for the test case
        test_case_prompt = self.generate_test_case_prompt(
            document=document
        )

        ##########
        # Final Prompt
        ##########
 
        # Combine the prompt parts
        prompt = self.prompt_template.format(
            entity_types_prompt=self.entity_types_prompt,
            demonstrations_prompt=demonstrations_prompt,
            contexts_prompt=contexts_prompt,
            test_case_prompt=test_case_prompt
        )

        return prompt

    def generate_demonstrations_prompt(
        self,
        demonstration_documents: list[Document]
    ) -> str:
        """Generate a prompt for the demonstrations."""

        prompt = ""
        n_demos = len(demonstration_documents)
        for demo_i, demo_doc in enumerate(demonstration_documents):
            prompt += f"Example {demo_i+1}:\n"
            prompt += f"Text: {self.generate_input_text_prompt(document=demo_doc)}\n"
            prompt += "Output:\n"
            prompt += f"{self.generate_output_prompt(document=demo_doc)}\n"
            if demo_i < n_demos - 1:
                prompt += "\n"

        return prompt.rstrip()
        
    def generate_contexts_prompt(self, context_texts: list[str]) -> str:
        """Generate a prompt for the contexts."""

        n_contexts = len(context_texts)
        if n_contexts == 0:
            return ""
        prompt = ""
        for context_i, content in enumerate(context_texts):
            prompt += f"[{context_i+1}] {content.strip()} \n"
            if context_i < n_contexts - 1:
                prompt += "\n"

        return prompt.rstrip()

    def generate_test_case_prompt(self, document: Document) -> str:
        """Generate a prompt for the test case."""

        prompt = f"Text: {self.generate_input_text_prompt(document=document)}\n"

        return prompt.rstrip()

    def generate_input_text_prompt(self, document: Document) -> str:
        """Generate a prompt for the input text."""

        prompt = " ".join(document["sentences"]) + "\n"

        return prompt.rstrip()

    def generate_output_prompt(self, document: Document) -> str:
        """Generate a prompt for the output mentions."""

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
        paths["path_snapshot"] = self.base_output_path

        # evaluation outputs
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        return paths

    def setup_dataset(
        self,
        extractor: LLMNER,
        documents: list[Document],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            gold_documents = []
            for document in tqdm(documents, desc="dataset setup"):
                gold_doc = copy.deepcopy(document)
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def save_extractor(self, extractor: LLMNER) -> None:
        extractor.save(path_snapshot=self.paths["path_snapshot"])

    def evaluate(
        self,
        extractor: LLMNER,
        documents: list[Document],
        contexts: list[ContextsForOneExample] | None,
        split: str,
        #
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:

        # Apply the extractor
        result_documents = extractor.batch_extract(
            documents=documents,
            contexts=contexts
        )

        # Save the prediction results
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)

        # Save the prompt-response pairs in plain text
        with open(
            self.paths[f"path_{split}_pred"].replace(".json", ".txt"), "w"
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
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"]
        )

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
