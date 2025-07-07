import copy
# import json
import logging
import os
import re
import unicodedata

# import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm

from ..models import LLM
from ..models import OpenAILLM
from .. import evaluation
from .. import utils


logger = logging.getLogger(__name__)


class LLMNER:

    def __init__(
        self,
        # General
        device,
        config,
        # Task specific
        vocab_etype,
        etype_meta_info,
        # Optional: few-shot setting
        path_demonstration_pool=None,
        # Misc.
        model=None,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        vocab_etype: dict[str, int] | str
        etype_meta_info: dict[str, dict[str, str]] | str
        path_demonstration_pool: str | None
            by default None
        model : LLM | OpenAILLM | None
            by default None
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> LLMNER Initialization >>>>>>>>>>")
        self.device = device

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
        # Vocabulary (entity types)
        ######

        if isinstance(vocab_etype, str):
            tmp = vocab_etype
            vocab_etype = utils.read_vocab(vocab_etype)
            if self.verbose:
                logger.info(f"Loaded entity type vocabulary from {tmp}")
        self.vocab_etype = vocab_etype
        self.ivocab_etype = {i:l for l, i in self.vocab_etype.items()}

        if isinstance(etype_meta_info, str):
            tmp = etype_meta_info
            etype_meta_info = utils.read_json(etype_meta_info)
            if self.verbose:
                logger.info(f"Loaded entity type meta-information from {tmp}")
        self.etype_meta_info = etype_meta_info

        ######
        # Prompt Processor
        ######

        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=config[
                "prompt_template_name_or_path"
            ],
            vocab_etype=self.vocab_etype,
            etype_meta_info=self.etype_meta_info,
            path_demonstration_pool=path_demonstration_pool,
            n_demonstrations=config["n_demonstrations"]
        )

        ######
        # Model
        ######

        self.model_name = config["model_name"]
        assert self.model_name in ["llm", "openai"]

        if model is not None:
            self.model = model
            logger.info("LLM is provided")
        elif self.model_name == "llm":
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
                # Model
                openai_model_name=config["openai_model_name"],
                # Generation
                max_new_tokens=config["max_new_tokens"]
            )
        # self.model.llm.to(self.model.device)

        ######
        # Generated text parsing
        ######

        # Output Format:
        # <bullet> (<mention>, <entity type>)
        # self.re_comp = re.compile("(.+?)\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)$")
        # <bullet> <mention> -> <entity type>
        # self.re_comp = re.compile("(.+?)\s*(.+?)\s*->\s*(.+?)$")
        # <bullet> <mention> | <entity type>
        self.re_comp = re.compile("(.+?)\s*(.+?)\s*\|\s*(.+?)$")

        # self.normalized_to_canonical = {
        #     etype.lower(): etype
        #     for etype in list(self.vocab_etype.keys())
        # }
        # self.normalized_to_canonical = {
        #     pretty_etype.lower(): etype
        #     for etype, pretty_etype in config["etype_name_to_pretty_etype_name"].items()
        # }
        self.normalized_to_canonical = {}
        for etype in self.vocab_etype.keys():
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            self.normalized_to_canonical[pretty_name.lower()] = etype

        if self.verbose:
            logger.info("<<<<<<<<<< LLMNER Initialization <<<<<<<<<<")

    def extract(
        self,
        document,
        # optional: few-shot setting
        demonstrations_for_doc=None,
        # optional: context augmentation
        contexts_for_doc=None
    ):
        """
        Parameters
        ----------
        document : Document
        demonstrations_for_doc : dict[str, str | list[DemoKeyInfo]] | None
            by default None
        contexts_for_doc : dict[str, str | list[str]] | None
            by default None  

        Returns
        -------
        Document | list[Document]
        """
        with torch.no_grad():
            if self.model_name == "llm":
                # Switch to inference mode
                self.model.llm.eval()

            # Generate a prompt
            prompt = self.prompt_processor.generate(
                document=document,
                demonstrations_for_doc=demonstrations_for_doc,
                contexts_for_doc=contexts_for_doc
            )

            if self.model_name == "llm":
                # Preprocess
                preprocessed_data = self.model.preprocess(prompt=prompt)

                # Tensorize
                model_input = self.model.tensorize(
                    preprocessed_data=preprocessed_data,
                    compute_loss=False
                )

                # Forward
                generated_text = self.model.generate(**model_input)[0] # str
                generated_text = self.model.remove_prompt_from_generated_text(
                    generated_text=generated_text
                )
            else:
                generated_text = self.model.generate(prompt) # str

            # Structurize
            mentions = self.structurize(
                document=document,
                generated_text=generated_text
            )

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["mentions"] = mentions
            if self.model_name == "llm":
                result_document["ner_prompt"] = preprocessed_data["prompt"]
            else:
                result_document["ner_prompt"] = prompt
            result_document["ner_generated_text"] = generated_text
            return result_document

    def structurize(self, document, generated_text):
        """
        Parameters
        ----------
        generated_text : str
        document : Document

        Returns
        -------
        list[Mention]
        """
        doc_key = document["doc_key"]

        # Get mapping from character position to word position (index)
        original_words = " ".join(document["sentences"]).split()
        normalized_words = [unicodedata.normalize("NFC", w).lower() for w in original_words]        
        normalized_text = " ".join(normalized_words)
       
        # NOTE: char_index_to_word_index must be created based on the normalized (lowered) text 
        char_index_to_word_index = [] # list[int]
        for w_i, w in enumerate(normalized_words):
            # n_chars = len(w)
            # char_index_to_word_index.extend([w_i] * n_chars)
            # char_index_to_word_index.append(None) # for space between words
            if w_i > 0:
                char_index_to_word_index.append(None) # space
            for _ in w:
                char_index_to_word_index.append(w_i)

        # Parse each generated line
        generated_lines = generated_text.split("\n")
        tuples = [] # list[tuple[int, int, str]]
        token_index_to_sent_index = [] # list[int]
        for s_i, sent in enumerate(document["sentences"]):
            s_len = len(sent.split())
            token_index_to_sent_index.extend([s_i] * s_len)
        for generated_line in generated_lines:
            generated_line = generated_line.strip()
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
            spans = [(b,e) for b,e in spans if token_index_to_sent_index[b] == token_index_to_sent_index[e]]
            # Remove too-long spans
            spans = [(b,e) for b,e in spans if (e - b) <= 10]
            if len(spans) == 0:
                logger.info(f"[{doc_key}] Skipped a generated line with invalid mention: '{generated_line}'")
                continue

            # Check whether the entity type can be found in the possible list
            normalized_entity_type = entity_type.lower()
            if not normalized_entity_type in self.normalized_to_canonical:
                logger.info(f"[{doc_key}] A generated line contains invalid entity type: '{generated_line}'")
                # continue
            # canonical_entity_type \
            #     = self.normalized_to_canonical[normalized_entity_type]
            canonical_entity_type = self.normalized_to_canonical.get(normalized_entity_type, entity_type)

            # Add new tuples
            for begin_token_i, end_token_i in spans:
                tuple_ = (
                    begin_token_i,
                    end_token_i,
                    canonical_entity_type
                )
                if not tuple_ in tuples:
                    tuples.append(tuple_)

        # Convert tuples to mention dicts
        mentions = [] # list[Mention]
        for (begin_i, end_i, etype) in tuples:
            name = " ".join(original_words[begin_i: end_i + 1])
            mentions.append({
                "span": (begin_i, end_i),
                "name": name,
                "entity_type": etype,
            })
        mentions = sorted(mentions, key=lambda m: m["span"])
        return mentions

    # def structurize(self, document, generated_text):
    #     """
    #     Parameters
    #     ----------
    #     generated_text : str
    #     document : Document

    #     Returns
    #     -------
    #     list[Mention]
    #     """
    #     # Get mapping from character position to word position (index)
    #     # words = utils.flatten_lists([s.split() for s in document["sentences"]])
    #     words = " ".join(document["sentences"]).split()
    #     normalized_text = " ".join(words).lower()
    #     char_index_to_word_index = [] # list[int]
    #     for w_i, w in enumerate(words):
    #         n_chars = len(w)
    #         char_index_to_word_index.extend([w_i] * n_chars)
    #         char_index_to_word_index.append(None) # for space between words

    #     # Parse the generated text into a JSON object
    #     begin_index = generated_text.find("{")
    #     end_index = generated_text.rfind("}")
    #     if begin_index < 0 or end_index < 0:
    #         logger.info(f"Skipped extraction because we could not parse the generated text into a JSON object: '{generated_text}'")
    #         return []
    #     json_text = generated_text[begin_index: end_index + 1]
    #     try:
    #         json_obj = json.loads(json_text)
    #     except Exception as e:
    #         logger.info(f"Skipped extraction because we could not parse the generated text into a JSON object: '{generated_text}'")
    #         logger.info(e)
    #         return []
    #     if not isinstance(json_obj, dict):
    #         logger.info(f"Skipped extraction because the parsed JSON object is not a dictionary: '{json_obj}'")
    #         return []
    #     if not "mentions" in json_obj:
    #         logger.info(f"Skipped extraction because the parsed JSON object does not contain `mentions' key: '{json_obj}'")
    #         return []

    #     # We process each entry in the list
    #     tuples = [] # list[tuple[int, int, str]]
    #     for entry in json_obj["mentions"]:
    #         if (not "mention" in entry) or (not "entity_type" in entry):
    #             logger.info(f"Skipped a parsed JSON entry of invalid formatting: '{entry}'")
    #             continue
    #         name = entry["mention"]
    #         entity_type = entry["entity_type"]

    #         # Check whether the mention can be found in the input text
    #         # i.e., get word-level spans
    #         normalized_name = name.lower()
    #         spans = self.extract_word_level_spans(
    #             normalized_name=normalized_name,
    #             normalized_text=normalized_text,
    #             char_index_to_word_index=char_index_to_word_index
    #         )
    #         if len(spans) == 0:
    #             logger.info(f"Skipped a parsed JSON entry with invalid mention: '{entry}'")
    #             continue

    #         # Check whether the entity type can be found in the possible list
    #         normalized_entity_type = entity_type.lower()
    #         if not normalized_entity_type in self.normalized_to_canonical:
    #             logger.info(f"Skipped a parsed JSON entry with invalid entity type: '{entry}'")
    #             continue
    #         canonical_entity_type \
    #             = self.normalized_to_canonical[normalized_entity_type]

    #         # Add new tuples
    #         for begin_token_i, end_token_i in spans:
    #             tuple_ = (
    #                 begin_token_i,
    #                 end_token_i,
    #                 canonical_entity_type
    #             )
    #             if not tuple_ in tuples:
    #                 tuples.append(tuple_)

    #     # Convert tuples to mention dicts
    #     mentions = [] # list[Mention]
    #     for tuple_ in tuples:
    #         begin_i, end_i, etype = tuple_
    #         name = " ".join(words[begin_i: end_i + 1])
    #         mentions.append({
    #             "span": (begin_i, end_i),
    #             "name": name,
    #             "entity_type": etype,
    #         })
    #     mentions = sorted(mentions, key=lambda m: m["span"])
    #     return mentions

    def extract_word_level_spans(
        self,
        normalized_name,
        normalized_text,
        char_index_to_word_index
    ):
        """
        Parameters
        ----------
        normalized_name : str
        normalized_text : str
        char_index_to_word_index : list[int]

        Returns
        -------
        list[tuple[int,int]]
        """
        spans = [] # list[tuple[int,int]]
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
        documents,
        # optional: few-shot setting
        demonstrations=None,
        # optional: context augmentation
        contexts=None
    ):
        """
        Parameters
        ----------
        documents : list[Document]
        demonstrations : list[dict[str, str | list[DemoKeyInfo]]] | None
        contexts : list[dict[str, str | list[str]]] | None

        Returns
        -------
        list[Document]
        """
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
        prompt_template_name_or_path,
        vocab_etype,
        etype_meta_info,
        # optional: few-shot setting
        path_demonstration_pool=None,
        n_demonstrations=None
    ):
        """
        Parameters
        ----------
        prompt_template_name : str
        vocab_etype: dict[str, int]
        etype_meta_info: dict[str, dict[str, str]]
        path_demonstration_pool : str | None
            by default None
        n_demonstrations : int | None
            by default None
        """
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.vocab_etype = vocab_etype
        self.etype_meta_info = etype_meta_info
        self.path_demonstration_pool = path_demonstration_pool
        self.n_demonstrations = n_demonstrations

        if self.path_demonstration_pool is not None:
            assert self.n_demonstrations is not None

        #####
        # Load prompt template
        #####

        self.prompt_template = utils.read_prompt_template(prompt_template_name_or_path=self.prompt_template_name_or_path)
 
        # Check requirements
        assert "{entity_types_prompt}" in self.prompt_template
        if self.path_demonstration_pool is not None:
            assert "{demonstrations_prompt}" in self.prompt_template
        # assert "{contexts_prompt}" in self.prompt_template
        assert "{task_prompt}" in self.prompt_template

        self.entity_types_prompt = ""
        for etype in vocab_etype.keys():
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            definition = self.etype_meta_info[etype]["Definition"]
            self.entity_types_prompt += f"- {pretty_name}: {definition}\n"
        self.entity_types_prompt = self.entity_types_prompt.rstrip()

        #####
        # Load demonstration pool
        #####

        if self.path_demonstration_pool is not None:
            # dict[DocKey, Document]
            self.demonstration_pool = {
                demo_doc["doc_key"]: demo_doc
                for demo_doc in utils.read_json(self.path_demonstration_pool)
            }

    def generate(
        self,
        document,
        # optional: few-shot setting
        demonstrations_for_doc=None,
        # optional: context augmentation
        contexts_for_doc=None
    ):
        """
        Parameters
        ----------
        document : Document
        demonstrations_for_doc : dict[str, str | list[DemoKeyInfo]] | None
            by default None
        contexts_for_doc : dict[str, str | list[Passage]] | None
            by default None

        Returns
        -------
        str
        """
        if demonstrations_for_doc is not None:
            # Prepare demonstrations
            demonstration_documents = [] # list[Document]
            for demo_key_dict in (
                demonstrations_for_doc["demonstrations"][:self.n_demonstrations]
            ):
                demo_doc = self.demonstration_pool[demo_key_dict["doc_key"]]
                demonstration_documents.append(demo_doc)
            # Get prompt part for demonstrations
            demonstrations_prompt = self.generate_demonstrations_prompt(
                demonstration_documents=demonstration_documents
            )
        else:
            demonstrations_prompt = ""

        if contexts_for_doc is not None:
            # Prepare contexts
            context_texts = [] # list[str]
            for passage in contexts_for_doc["contexts"]:
                text = utils.create_text_from_passage(passage=passage, sep=" : ")
                context_texts.append(text)
            # Get prompt part for contexts
            contexts_prompt = self.generate_contexts_prompt(
                context_texts=context_texts
            )
        else:
            contexts_prompt = ""

        # Get prompt part for task
        task_prompt = self.generate_task_prompt(
            document=document
        )

        # Combine the prompt parts
        prompt = self.prompt_template.format(
            entity_types_prompt=self.entity_types_prompt,
            demonstrations_prompt=demonstrations_prompt,
            contexts_prompt=contexts_prompt,
            task_prompt=task_prompt
        )
        return prompt

    #####

    def generate_demonstrations_prompt(self, demonstration_documents):
        """
        Parameters
        ----------
        demonstration_documents: list[Document]

        Returns
        -------
        str
        """
        text = ""
        n_demos = len(demonstration_documents)
        for demo_i, demo_doc in enumerate(demonstration_documents):
            # Title
            text += f"Example {demo_i+1}:\n"
            # Input
            text += (
                "Text: "
                + self.generate_input_text_prompt(document=demo_doc)
                + "\n"
            )
            # Output
            text += (
                "Output:\n"
                + self.generate_output_prompt(document=demo_doc)
                + "\n"
            )
            if demo_i < n_demos - 1:
                text += "\n"
        return text.rstrip()
        
    def generate_contexts_prompt(self, context_texts):
        """
        Parameters
        ----------
        context_texts : list[str]

        Returns
        -------
        str
        """
        n_contexts = len(context_texts)
        if n_contexts == 0:
            return ""
        else:
            text = ""
            for context_i, content in enumerate(context_texts):
                text += f"[{context_i+1}] {content.strip()} \n"
                if context_i < n_contexts - 1:
                    text += "\n"
            return text.rstrip()

    def generate_task_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        str
        """
        text = ""
        # Input
        text += (
            "Text: "
            + self.generate_input_text_prompt(document=document)
            + "\n"
        )
        return text.rstrip()

    #####

    def generate_input_text_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        str
        """
        text = " ".join(document["sentences"]) + "\n"
        return text.rstrip()

    def generate_output_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        str
        """
        # names = []
        # entity_types = []
        # words = " ".join(document["sentences"]).split()
        # for m_i, mention in enumerate(document["mentions"]):
        #     begin_i, end_i = mention["span"]
        #     name = " ".join(words[begin_i: end_i + 1])
        #     if name in names:
        #         continue
        #     entity_type = mention["entity_type"]
        #     names.append(name)
        #     entity_types.append(entity_type)
        # text = ""
        # for name, entity_type in zip(names, entity_types):
        #     text += f"- {name} -> {entity_type}\n"
        # return text.rstrip()

        # output_json = {
        #     "mentions": []
        # }
        text = ""
        words = " ".join(document["sentences"]).split()
        for mention in document["mentions"]:
            begin_i, end_i = mention["span"]
            name = " ".join(words[begin_i: end_i + 1])
            etype = mention["entity_type"]
            pretty_name = self.etype_meta_info[etype]["Pretty Name"]
            # output_json["mentions"].append(
            #     {
            #         "mention": name,
            #         "entity_type": pretty_name
            #     }
            # )
            text += f"- {name} | {pretty_name}\n"
        # return json.dumps(output_json)
        
        # text = ""
        # for m in output_json["mentions"]:
        #     name = m["mention"]
        #     etype = m["entity_type"]
        #     text += f"- {name} | {etype}\n"
        # return text.rstrip()

        return text.rstrip()


class LLMNERTrainer:

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
        paths["path_vocab_etype"] = self.base_output_path + "/entity_types.vocab.txt"
        paths["path_etype_meta_info"] = self.base_output_path + "/etype_meta_info.json"

        # Paths to validation outputs and scores
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"

        # Paths to evaluation outputs and scores
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        return paths

    def setup_dataset(self, extractor, documents, split):
        """
        Parameters
        ----------
        extractor : LLMNER
        documents : list[Document]
        split : str
        """
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            gold_documents = []
            for document in tqdm(documents, desc="dataset setup"):
                gold_doc = copy.deepcopy(document)
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def save_extractor(self, extractor):
        """
        Parameters
        ----------
        extractor : LLMNER
        """
        # Since we do not finetune the model, we save the configuration and
        #   the entity type vocabulary by this function.

        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], extractor.config)
        utils.write_json(self.paths["path_config"], extractor.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

        # Save the vocabulary (only once)
        utils.write_vocab(self.paths["path_vocab_etype"], extractor.vocab_etype, write_frequency=False)
        logger.info("Saved entity type vocabulary to %s" % self.paths["path_vocab_etype"])

        utils.write_json(self.paths["path_etype_meta_info"], extractor.etype_meta_info)
        logger.info("Saved entity type meta-information to %s" % self.paths["path_etype_meta_info"])

    def evaluate(
        self,
        extractor,
        documents,
        demonstrations,
        contexts,
        split,
        #
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        extractor : LLMNER
        documents : list[Document]
        demonstrations : list[dict[str, str | list[DemoKeyInfo]]] | None
        contexts : list[dict[str, str | list[Passage]]] | None
        split : str
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # (documents, demonstrations, context) -> path_pred
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
        # (path_pred, path_gold) -> scores
        scores = evaluation.ner.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"]
        )
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
