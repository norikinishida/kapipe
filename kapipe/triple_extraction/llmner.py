from __future__ import annotations

import copy
# import json
import logging
import os
import re
from typing import Any
import unicodedata

# import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm

from ..models import LLM, OpenAILLM
from .. import evaluation
from .. import utils
from ..datatypes import (
    Config,
    Document,
    Mention,
    DemonstrationsForOneExample,
    ContextsForOneExample
)


logger = logging.getLogger(__name__)


class LLMNER:

    def __init__(
        self,
        device: str,
        # Initialization
        config: Config | str | None = None,
        vocab_etype: dict[str, int] | str | None = None,
        etype_meta_info: dict[str, dict[str, str]] | str | None = None,
        path_demonstration_pool: str | None = None,
        # Loading
        path_snapshot: str | None = None,
        # Misc
        model: LLM | OpenAILLM | None = None,
    ):
        logger.info("########## LLMNER Initialization Starts ##########")

        self.device = device
        self.path_snapshot = path_snapshot

        if path_snapshot is not None:
            assert config is None
            assert vocab_etype is None
            assert etype_meta_info is None
            assert path_demonstration_pool is None
            config = path_snapshot + "/config"
            vocab_etype = path_snapshot + "/entity_types.vocab.txt"
            etype_meta_info = path_snapshot + "/etype_meta_info.json"
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

        # Load the entity type vocabulary
        if isinstance(vocab_etype, str):
            vocab_path = vocab_etype
            vocab_etype = utils.read_vocab(vocab_path)
            logger.info(f"Loaded entity type vocabulary from {vocab_path}")
        self.vocab_etype = vocab_etype
        self.ivocab_etype = {i:l for l, i in self.vocab_etype.items()}

        # Load the entity type meta information (pretty names and definitions)
        if isinstance(etype_meta_info, str):
            meta_path = etype_meta_info
            etype_meta_info = utils.read_json(meta_path)
            logger.info(f"Loaded entity type meta-information from {meta_path}")
        self.etype_meta_info = etype_meta_info

        # Initialize the prompt processor
        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            vocab_etype=self.vocab_etype,
            etype_meta_info=self.etype_meta_info,
            path_demonstration_pool=path_demonstration_pool,
            n_demonstrations=config["n_demonstrations"]
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
        # <bullet> (<mention>, <entity type>)
        # self.re_comp = re.compile("(.+?)\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)$")
        # <bullet> <mention> -> <entity type>
        # self.re_comp = re.compile("(.+?)\s*(.+?)\s*->\s*(.+?)$")
        # <bullet> <mention> | <entity type>
        self.re_comp = re.compile("(.+?)\s*(.+?)\s*\|\s*(.+?)$")

        # Create entity type mapping (normalized pretty name -> canonical name)
        # e.g., "Location" -> "LOC"
        self.normalized_to_canonical = {}
        for etype in self.vocab_etype.keys():
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            self.normalized_to_canonical[pretty_name.lower()] = etype

        logger.info("########## LLMNER Initialization Ends ##########")

    def save(self, path_snapshot: str) -> None:
        path_config = path_snapshot + "/config"
        path_vocab = path_snapshot + "/entity_types.vocab.txt"
        path_meta_info = path_snapshot + "/etype_meta_info.json"
        path_demonstration_pool = path_snapshot + "/demonstration_pool.json"
        utils.write_json(path_config, self.config)
        utils.write_vocab(path_vocab, self.vocab_etype, write_frequency=False)
        utils.write_json(path_meta_info, self.etype_meta_info)
        if self.prompt_processor.path_demonstration_pool is not None:
            utils.write_json(
                path_demonstration_pool,
                list(self.prompt_processor.demonstration_pool.values())
            )

    def extract(
        self,
        document: Document,
        # optional: few-shot setting
        demonstrations_for_doc: DemonstrationsForOneExample | None = None,
        # optional: context augmentation
        contexts_for_doc: ContextsForOneExample | None = None
    ) -> Document:
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
                # Preprocess
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
            mentions = self.structurize(
                document=document,
                generated_text=generated_text
            )

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["mentions"] = mentions
            if self.model_name == "hf":
                result_document["ner_prompt"] = preprocessed_data["prompt"]
            else:
                result_document["ner_prompt"] = prompt
            result_document["ner_generated_text"] = generated_text
            return result_document

    def structurize(self, document: Document, generated_text: str) -> list[Mention]:
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
        vocab_etype: dict[str, int],
        etype_meta_info: dict[str, dict[str, str]],
        # optional: few-shot setting
        path_demonstration_pool: str | None = None,
        n_demonstrations: int | None = None
    ):
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.vocab_etype = vocab_etype
        self.etype_meta_info = etype_meta_info
        self.path_demonstration_pool = path_demonstration_pool
        self.n_demonstrations = n_demonstrations

        # If demonstration pool is provided, `n_demonstartions` should also be set
        if self.path_demonstration_pool is not None:
            assert self.n_demonstrations is not None

        # Load the prompt template
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path
        )

        # Generate the prompt part for entity types
        self.entity_types_prompt = ""
        for etype in vocab_etype.keys():
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            definition = self.etype_meta_info[etype]["Definition"]
            self.entity_types_prompt += f"- {pretty_name}: {definition}\n"
        self.entity_types_prompt = self.entity_types_prompt.rstrip()

        # Load the demonstration pool
        if self.path_demonstration_pool is not None:
            self.demonstration_pool: dict[str, Document] = {
                demo_doc["doc_key"]: demo_doc
                for demo_doc in utils.read_json(self.path_demonstration_pool)
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

        # Generate prompt part for the test case
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
        prompt = f"Text: {self.generate_input_text_prompt(document=document)}\n"
        return prompt.rstrip()

    def generate_input_text_prompt(self, document: Document) -> str:
        prompt = " ".join(document["sentences"]) + "\n"
        return prompt.rstrip()

    def generate_output_prompt(self, document: Document) -> str:
        prompt = ""
        words = " ".join(document["sentences"]).split()
        for mention in document["mentions"]:
            begin_i, end_i = mention["span"]
            name = " ".join(words[begin_i: end_i + 1])
            etype = mention["entity_type"]
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            prompt += f"- {name} | {pretty_name}\n"
        return prompt.rstrip()


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
        demonstrations: list[DemonstrationsForOneExample] | None,
        contexts: list[ContextsForOneExample] | None,
        split: str,
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
                prompt = result_doc["ner_prompt"]
                generated_text = result_doc["ner_generated_text"]
                f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                f.write(prompt + "\n\n")
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
