from collections import defaultdict
import copy
# import json
import logging
import os
import re

# import numpy as np
import torch
from tqdm import tqdm

from ..models import LLM
from ..models import OpenAILLM
from .. import evaluation
from .. import utils


logger = logging.getLogger(__name__)


N_CAND = 3


class LLMED:

    def __init__(
        self,
        # General
        device,
        config,
        # Task specific
        path_entity_dict,
        # Optional: few-shot setting
        path_demonstration_pool=None,
        path_candidate_entities_pool=None,
        # Misc.
        model=None,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        path_entity_dict: str
        path_demonstration_pool: str | None
            by default None
        path_candidate_entities_pool: str | None
            by default None
        model : LLM | None
            by default None
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> LLMED Initialization >>>>>>>>>>")
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
        # Prompt Processor
        ######

        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            knowledge_base_name_prompt=config["knowledge_base_name"],
            path_entity_dict=path_entity_dict,
            path_demonstration_pool=path_demonstration_pool,
            path_candidate_entities_pool=path_candidate_entities_pool,
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

        # Output format
        # <bullet> (<mention>, <entity id>)
        # self.re_comp = re.compile("(.+?)\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)$")
        # <bullet> <mention> -> <entity id>
        # self.re_comp = re.compile("(.+?)\s*(.+?)\s*->\s*(.+?)$")
        # <bullet> <mention> | <entity id>
        self.re_comp = re.compile("(.+?)\s*(.+?)\s*\|\s*(.+?)$")

        if self.verbose:
            logger.info("<<<<<<<<<< LLMED Initialization <<<<<<<<<<")

    def extract(
        self,
        document,
        candidate_entities_for_doc,
        # optional: few-shot setting
        demonstrations_for_doc=None,
        # optional: prompt augmentation
        contexts_for_doc=None
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, str | list[list[CandEntKeyInfo]]]
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
 
            # Run multiple rounds
            prompt_list = []
            generated_text_list = []
            target_mentions_list = []
            N_MENT_PER_CHUNK = 5
            indices = list(range(0, len(document["mentions"])))
            for m_i in range(0, len(document["mentions"]), N_MENT_PER_CHUNK):
                target_mention_indices = indices[m_i: m_i + N_MENT_PER_CHUNK]

                # Generate a prompt
                prompt = self.prompt_processor.generate(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                    target_mention_indices=target_mention_indices,
                    demonstrations_for_doc=demonstrations_for_doc,
                    contexts_for_doc=contexts_for_doc
                )

                if self.model_name == "llm":
                    # Preprocess
                    preprocessed_data = self.model.preprocess(prompt=prompt)
                    prompt_list.append(preprocessed_data["prompt"])

                    if preprocessed_data["skip"]:
                        logger.info(f"Skipped generation because the text length ({sum([len(seg) for seg in preprocessed_data['llm_input']['segments_id']])}) is too long.")
                        generated_text = "SKIPPED BECAUSE THE TEXT IS TOO LONG"
                    else:
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
                    prompt_list.append(prompt)
                    generated_text = self.model.generate(prompt) # str
                
                generated_text_list.append(generated_text)

                # Structurize (1)
                target_mentions = self._structurize_for_mentions(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                    generated_text=generated_text,
                    target_mention_indices=target_mention_indices
                )
                target_mentions_list.append(target_mentions)

            # Structurize (2)
            mentions = utils.flatten_lists(target_mentions_list)
            assert len(mentions) == len(document["mentions"])
            entities = utils.aggregate_mentions_to_entities(
                document=document,
                mentions=mentions
            )

            # Integrate
            result_document = copy.deepcopy(document)
            for m_i in range(len(result_document["mentions"])):
                result_document["mentions"][m_i].update(mentions[m_i])
            result_document["entities"] = entities
            result_document["ed_prompt"] = "\n@@@@@@@@@@\n".join(prompt_list)
            result_document["ed_generated_text"] = "\n@@@@@@@@@@\n".join(generated_text_list)
            return result_document

    def _structurize_for_mentions(
        self,
        document,
        candidate_entities_for_doc,
        generated_text,
        target_mention_indices
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, str | list[list[CandEntKeyInfo]]]
        generated_text : str
        target_mention_indices : list[int]

        Returns
        -------
        list[Mention]
        """
        doc_key = document["doc_key"]

        # Get one-to-many mapping from normalized mention name to mention indices
        normalized_name_to_mention_indices = defaultdict(list)
        # words = utils.flatten_lists([s.split() for s in document["sentences"]])
        words = " ".join(document["sentences"]).split()
        for m_i, mention in enumerate(document["mentions"]):
            if m_i in target_mention_indices:
                b_i, e_i = mention["span"]
                name = " ".join(words[b_i: e_i+1])
                normalized_name = name.lower()
                normalized_name_to_mention_indices[normalized_name].append(m_i)
        
        # Get a list of possible entity IDs
        possible_entity_ids = []
        for cands in candidate_entities_for_doc["candidate_entities"]:
            for cand in cands:
                possible_entity_ids.append(cand["entity_id"])
        possible_entity_ids = set(possible_entity_ids)

        # Initialize the output mentions
        # NOTE: We create a list of mentions, whose length is the same with the original number of mentions
        # We will filter out the mentions later using the target_mention_indices
        mentions = []
        for _ in range(len(document["mentions"])):
            mentions.append(
                {
                    "entity_id": "NO-PRED",
                    # "corresponding_output_line": None
                }
            )

        # Parse each generated line
        names = []
        entity_ids = []
        generated_lines = generated_text.split("\n")
        for generated_line in generated_lines:
            generated_line = generated_line.strip()
            if generated_line == "":
                continue

            # Parse the generated line
            parsed = self.re_comp.findall(generated_line)
            if not (len(parsed) == 1 and len(parsed[0]) == 3):
                logger.info(f"[{doc_key}] Skipped a generated line of invalid formatting: '{generated_line}'")
                continue
            _, name, entity_id = parsed[0]
            names.append(name)
            entity_ids.append(entity_id)

        if len(names) == len(entity_ids) == len(document["mentions"]):
            # The number of entities (i.e., len(names), len(entity_ids)) is the same with that of all mentions in the document
            if len(document["mentions"]) == 0:
                assert len(target_mention_indices) == 0
            else:
                for m_i, (name, entity_id) in enumerate(zip(names, entity_ids)):
                    # Skip cheking for the mention names
    
                    # Check whether the entity ID can be found in the possible list
                    if not entity_id in possible_entity_ids:
                        logger.info(f"[{doc_key}] Skipped a generated line with invalid concept ID: {entity_id}")
                        continue
    
                    # Add mention
                    mentions[target_mention_indices[m_i]]["entity_id"] = entity_id

        else:
            for name, entity_id in zip(names, entity_ids):

                # Check whether the mention can be found in the possible list
                normalized_name = name.lower()
                # if not normalized_name in normalized_name_to_mention_indices:
                #     logger.info(f"Skipped a parsed JSON entry with invalid mention: '{entry}' not in {list(normalized_name_to_mention_indices.keys())}")
                #     continue
                pattern = r"\s*".join(re.escape(c) for c in normalized_name)
                normalized_name2 = None
                for n in normalized_name_to_mention_indices.keys():
                    results = list(re.finditer(
                        "@@@" + pattern + "@@@",
                        "@@@" + n + "@@@"
                    ))
                    if len(results) == 1:
                        normalized_name2 = n
                        break
                if normalized_name2 is None:
                    logger.info(f"[{doc_key}] Skipped a generated line with invalid mention: '{normalized_name}' not in {list(normalized_name_to_mention_indices.keys())}")
                    continue
                normalized_name = normalized_name2

                # Check whether the entity ID can be found in the possible list
                if not entity_id in possible_entity_ids:
                    logger.info(f"[{doc_key}] Skipped a generated line with invalid concept ID: {entity_id}")
                    continue

                # Add mention
                mention_indices = normalized_name_to_mention_indices[normalized_name]
                for m_i in mention_indices:
                    mentions[m_i]["entity_id"] = entity_id

        # Check
        for m_i in range(len(document["mentions"])):
            if not m_i in target_mention_indices:
                assert mentions[m_i]["entity_id"] == "NO-PRED"

        return [mentions[m_i] for m_i in target_mention_indices]

    # def _structurize_for_mentions(
    #     self,
    #     document,
    #     candidate_entities_for_doc,
    #     generated_text,
    #     target_mention_indices
    # ):
    #     """
    #     Parameters
    #     ----------
    #     document : Document
    #     candidate_entities_for_doc : dict[str, str | list[list[CandEntKeyInfo]]]
    #     generated_text : str
    #     target_mention_indices : list[int]

    #     Returns
    #     -------
    #     list[Mention]
    #     """
    #     # Get one-to-many mapping from normalized mention name to mention indices
    #     normalized_name_to_mention_indices = defaultdict(list)
    #     # words = utils.flatten_lists([s.split() for s in document["sentences"]])
    #     words = " ".join(document["sentences"]).split()
    #     for m_i, mention in enumerate(document["mentions"]):
    #         if m_i in target_mention_indices:
    #             b_i, e_i = mention["span"]
    #             name = " ".join(words[b_i: e_i+1])
    #             normalized_name = name.lower()
    #             normalized_name_to_mention_indices[normalized_name].append(m_i)

    #     # Get a list of possible entity IDs
    #     possible_entity_ids = []
    #     for cands in candidate_entities_for_doc["candidate_entities"]:
    #         for cand in cands:
    #             possible_entity_ids.append(cand["entity_id"])
    #     possible_entity_ids = set(possible_entity_ids)

    #     # Initialize the output mentions
    #     # NOTE: We create a list of mentions, whose length is the same with the original number of mentions
    #     # We will filter out the mentions later using the target_mention_indices
    #     mentions = []
    #     for _ in range(len(document["mentions"])):
    #         mentions.append(
    #             {
    #                 "entity_id": "NO-PRED",
    #                 # "corresponding_output_line": None
    #             }
    #         )

    #     # Parse the generated text into a JSON object
    #     begin_index = generated_text.find("{")
    #     end_index = generated_text.rfind("}")
    #     if begin_index < 0 or end_index < 0:
    #         logger.info(f"Skipped extraction because we could not parse the generated text into a JSON object: '{generated_text}'")
    #         return [mentions[m_i] for m_i in target_mention_indices]
    #     json_text = generated_text[begin_index: end_index + 1]
    #     try:
    #         json_obj = json.loads(json_text)
    #     except Exception as e:
    #         logger.info(f"Skipped extraction because we could not parse the generated text into a JSON object: '{generated_text}'")
    #         logger.info(e)
    #         return [mentions[m_i] for m_i in target_mention_indices]
    #     if not isinstance(json_obj, dict):
    #         logger.info(f"Skipped extraction because the parsed JSON object is not a dictionary: '{json_obj}'")
    #         return [mentions[m_i] for m_i in target_mention_indices]
    #     if not "entity_disambiguation" in json_obj:
    #         logger.info(f"Skipped extraction because the parsed JSON object does not contain `entity_disambiguation' key: '{json_obj}'")
    #         return [mentions[m_i] for m_i in target_mention_indices]

    #     # We process each entry in the list
    #     if len(json_obj["entity_disambiguation"]) == len(target_mention_indices):

    #         for m_i, entry in enumerate(json_obj["entity_disambiguation"]):
    #             # NOTE: We do not care about the mention
    #             # name = entry["mention"]
    #             entity_id = entry["concept_id"]

    #             # Check whether the entity ID can be found in the possible list
    #             if not entity_id in possible_entity_ids:
    #                 logger.info(f"Skipped a parsed JSON entry with invalid concept ID: {entity_id}")
    #                 continue

    #             # Add mention
    #             mentions[target_mention_indices[m_i]]["entity_id"] = entity_id

    #     else:

    #         for entry in json_obj["entity_disambiguation"]:
    #             if (not "mention" in entry) or (not "concept_id" in entry):
    #                 logger.info(f"Skipped a parsed JSON entry of invalid formatting: '{entry}'")
    #                 continue
    #             name = entry["mention"]
    #             entity_id = entry["concept_id"]

    #             # Check whether the mention can be found in the possible list
    #             normalized_name = name.lower()
    #             # if not normalized_name in normalized_name_to_mention_indices:
    #             #     logger.info(f"Skipped a parsed JSON entry with invalid mention: '{entry}' not in {list(normalized_name_to_mention_indices.keys())}")
    #             #     continue
    #             pattern = r"\s*".join(re.escape(c) for c in normalized_name)
    #             normalized_name2 = None
    #             for n in normalized_name_to_mention_indices.keys():
    #                 results = list(re.finditer(
    #                     "@@@" + pattern + "@@@",
    #                     "@@@" + n + "@@@"
    #                 ))
    #                 if len(results) == 1:
    #                     normalized_name2 = n
    #                     break
    #             if normalized_name2 is None:
    #                 logger.info(f"Skipped a parsed JSON entry with invalid mention: '{entry}' not in {list(normalized_name_to_mention_indices.keys())}")
    #                 continue
    #             normalized_name = normalized_name2

    #             # Check whether the entity ID can be found in the possible list
    #             if not entity_id in possible_entity_ids:
    #                 logger.info(f"Skipped a parsed JSON entry with invalid concept ID: {entity_id}")
    #                 continue

    #             # Add mention
    #             mention_indices \
    #                 = normalized_name_to_mention_indices[normalized_name]
    #             for m_i in mention_indices:
    #                 mentions[m_i]["entity_id"] = entity_id

    #     # Check
    #     for m_i in range(len(document["mentions"])):
    #         if not m_i in target_mention_indices:
    #             assert mentions[m_i]["entity_id"] == "NO-PRED"

    #     return [mentions[m_i] for m_i in target_mention_indices]

    def batch_extract(
        self,
        documents,
        candidate_entities,
        # optional: few-shot setting
        demonstrations=None,
        # optional: context augmentation
        contexts=None
    ):
        """
        Parameters
        ----------
        documents : list[Document]
        candidate_entities: list[dict[str, str | list[list[CandEntKeyInfo]]]]
        demonstrations: list[dict[str, str | list[DemoKeyInfo]]] | None
            by default None
        contexts : list[dict[str, str | list[str]]] | None
            by default None

        Returns
        -------
        list[Document]
        """
        result_documents = []
        if demonstrations is None:
            demonstrations = [None] * len(documents)
        if contexts is None:
            contexts = [None] * len(documents)
        for (
            document,
            candidate_entities_for_doc,
            demonstrations_for_doc,
            contexts_for_doc
        ) in tqdm(
                zip(
                    documents,
                    candidate_entities,
                    demonstrations,
                    contexts
                ),
                total=len(documents),
                desc="extraction steps"
            ):
            result_document = self.extract(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc,
                demonstrations_for_doc=demonstrations_for_doc,
                contexts_for_doc=contexts_for_doc
            )
            result_documents.append(result_document)
        return result_documents


class PromptProcessor:

    def __init__(
        self,
        prompt_template_name_or_path,
        knowledge_base_name_prompt,
        path_entity_dict,
        # optional: few-shot setting
        path_demonstration_pool=None,
        path_candidate_entities_pool=None,
        n_demonstrations=None
    ):
        """
        Parameters
        ----------
        prompt_template_name : str
        knowledge_base_name_prompt : str
        path_entity_dict : str
        path_demonstration_pool : str | None
            by default None
        path_candidate_entities_pool : str | None
            by default None
        n_demonstrations : int | None
            by default None
        """
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.knowledge_base_name_prompt = knowledge_base_name_prompt
        self.path_entity_dict = path_entity_dict
        self.path_demonstration_pool = path_demonstration_pool
        self.path_candidate_entities_pool = path_candidate_entities_pool
        self.n_demonstrations = n_demonstrations

        if self.path_demonstration_pool is not None:
            assert self.path_candidate_entities_pool is not None
            assert self.n_demonstrations is not None

        #####
        # Load prompt template
        #####

        self.prompt_template = utils.read_prompt_template(prompt_template_name_or_path=self.prompt_template_name_or_path)
 
        # Check requirements
        assert "{knowledge_base_name_prompt}" in self.prompt_template
        if self.path_demonstration_pool is not None:
            assert "{demonstrations_prompt}" in self.prompt_template
        # assert "{contexts_prompt}" in self.prompt_template
        assert "{task_prompt}" in self.prompt_template

        #####
        # Load entity dictionary
        #####

        # dict[str, EntityPage]
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }

        #####
        # Load pools for demonstrations and candidate entities
        #####

        if self.path_demonstration_pool is not None:
            # dict[DocKey, Document]
            self.demonstration_pool = {
                demo_doc["doc_key"]: demo_doc
                for demo_doc in utils.read_json(path_demonstration_pool)
            }

            # dict[DocKey, dict[str, str | list[list[CandEntKeyInfo]]]]
            self.candidate_entities_pool = {
                cands["doc_key"]: cands
                for cands in utils.read_json(path_candidate_entities_pool)
            }

    def generate(
        self,
        document,
        candidate_entities_for_doc,
        target_mention_indices,
        # optional: few-shot setting
        demonstrations_for_doc=None,
        # optional: context augmentation
        contexts_for_doc=None
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, str | list[list[CandEntKeyInfo]]]
        target_mention_indices : list[int]
        demonstrations_for_doc : dict[str, str | list[DemoKeyInfo]] | None
            by default None
        contexts_for_doc : dict[str, str | list[Passage]] | None
            by default None
        
        Returns
        -------
        str
        """
        # Prepare candidate entities for the input document
        candidate_entity_pages_for_doc = [] # list[list[EntityPage]]
        for candidate_entities_for_one_mention in candidate_entities_for_doc["candidate_entities"]:
            candidate_entity_pages_for_one_mention = [
                self.entity_dict[cand_key_dict["entity_id"]]
                for cand_key_dict in candidate_entities_for_one_mention
            ]
            candidate_entity_pages_for_doc.append(candidate_entity_pages_for_one_mention)

        if demonstrations_for_doc is not None:
            # Prepare demonstration documents
            demonstration_documents = [] # list[Document]
            for demo_key_dict in demonstrations_for_doc["demonstrations"][:self.n_demonstrations]:
                demo_doc = self.demonstration_pool[demo_key_dict["doc_key"]]
                demonstration_documents.append(demo_doc)

            # Prepare candidate entities for the demonstration documents
            candidate_entity_pages_for_demos = [] # list[list[list[EntityPage]]]
            for demo_key_dict in demonstrations_for_doc["demonstrations"][:self.n_demonstrations]:
                candidate_entities_for_demo = self.candidate_entities_pool[demo_key_dict["doc_key"]]
                candidate_entity_pages_for_demo = [] # list[list[EntityPage]]
                for candidate_entities_for_one_mention in candidate_entities_for_demo["candidate_entities"]:
                    candidate_entity_pages_for_one_mention = [
                        self.entity_dict[cand_key_dict["entity_id"]]
                        for cand_key_dict in candidate_entities_for_one_mention
                    ]
                    candidate_entity_pages_for_demo.append(candidate_entity_pages_for_one_mention)
                candidate_entity_pages_for_demos.append(candidate_entity_pages_for_demo)

            # Get prompt part for demonstrations
            demonstrations_prompt = self.generate_demonstrations_prompt(
                demonstration_documents=demonstration_documents,
                candidate_entity_pages_for_demos=candidate_entity_pages_for_demos
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
            document=document,
            candidate_entity_pages_for_doc=candidate_entity_pages_for_doc,
            target_mention_indices=target_mention_indices
        )

        # Combine the prompt parts
        prompt = self.prompt_template.format(
            knowledge_base_name_prompt=self.knowledge_base_name_prompt,
            demonstrations_prompt=demonstrations_prompt,
            contexts_prompt=contexts_prompt,
            task_prompt=task_prompt
        )
        return prompt

    #####

    def generate_demonstrations_prompt(
        self,
        demonstration_documents,
        candidate_entity_pages_for_demos
    ):
        """
        Parameters
        ----------
        demonstration_documents: list[Document]
            shape of (n_demos,)
        candidate_entity_pages_for_demos: list[list[list[EntityPage]]]
            shape of (n_demos, n_mentions, n_candidates)

        Returns
        -------
        str
        """
        text = ""
        n_demos = len(demonstration_documents)
        for demo_i, (demo_doc, cand_ent_pages_for_demo) in enumerate(
            zip(
                demonstration_documents,
                candidate_entity_pages_for_demos
            )
        ):
            # Title
            text += f"Example {demo_i+1}:\n"
            # Input
            text += (
                "Text: "
                + self.generate_input_text_prompt(document=demo_doc)
                + "\n"
            )
            target_mention_indices = [
                m_i for m_i, m in enumerate(demo_doc["mentions"])
                if m["entity_id"] in self.entity_dict
            ]
            mention_candidates_pairs_prompt = self.generate_input_mention_candidates_pairs_prompt(
                document=demo_doc, 
                candidate_entity_pages_for_doc=cand_ent_pages_for_demo,
                target_mention_indices=target_mention_indices[:2],
                demonstration_mode=True
            )
            text += (
                # "Entity mentions and list of candidate concept IDs:\n"
                mention_candidates_pairs_prompt
                + "\n"
            )
            # Output
            text += (
                "Output:\n"
                + self.generate_output_prompt(
                    document=demo_doc,
                    candidate_entity_pages_for_doc=cand_ent_pages_for_demo,
                    target_mention_indices=target_mention_indices[:2],
                )
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

    def generate_task_prompt(
        self,
        document,
        candidate_entity_pages_for_doc,
        target_mention_indices
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entity_pages_for_doc : list[list[EntityPage]]
            shape of (n_mentions, n_candidates)
        target_mention_indices : list[int]
            shape of (n_target_mentions,)

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
        mention_candidates_pairs_prompt = self.generate_input_mention_candidates_pairs_prompt(
            document=document, 
            candidate_entity_pages_for_doc=candidate_entity_pages_for_doc,
            target_mention_indices=target_mention_indices
        )
        text += (
            # "Entity mentions and list of candidate concept IDs:\n"
            mention_candidates_pairs_prompt
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

    # def generate_mentions_prompt(self, document, demo=False):
    #     """
    #     Parameters
    #     ----------
    #     document : Document
    #     demo : bool
    #         by default False

    #     Returns
    #     -------
    #     tuple[str, list[int]]
    #     """
    #     text = ""
    #     words = " ".join(document["sentences"]).split()
    #     names = []
    #     selected_mention_indices = []
    #     for m_i, mention in enumerate(document["mentions"]):
    #         begin_i, end_i = mention["span"]
    #         name = " ".join(words[begin_i: end_i + 1])
    #         if name in names:
    #             continue
    #         names.append(name)
    #         selected_mention_indices.append(m_i)
    #         # In the demonstrations, we skip some mentions
    #         if demo and len(names) >= 3:
    #             break
    #     for n_i, name in enumerate(names):
    #         text += f"{n_i + 1}. {name}\n"
    #     return text.rstrip(), selected_mention_indices

    # def generate_candidate_entities_prompt(
    #     self,
    #     candidate_entity_pages_for_doc,
    #     selected_mention_indices=None
    # ):
    #     """
    #     Parameters
    #     ----------
    #     candidate_entity_pages_for_doc : list[list[EntityPage]]
    #     selected_mention_indices : list[int] | None
    #         by default None

    #     Returns
    #     -------
    #     str
    #     """
    #     N_CAND = 3
    #     # Aggregate candidates as a single list
    #     candidates = []
    #     memorized_ids = set()
    #     for m_i, candidate_entity_pages_for_one_mention in (
    #         enumerate(candidate_entity_pages_for_doc)
    #     ):
    #         if (
    #             (selected_mention_indices is not None)
    #             and
    #             (not m_i in selected_mention_indices)
    #         ):
    #             continue
    #         for cand_page in candidate_entity_pages_for_one_mention[:N_CAND]:
    #             if not cand_page["entity_id"] in memorized_ids:
    #                 candidates.append(cand_page)
    #                 memorized_ids.add(cand_page["entity_id"])
    #     # Transform the candidate list into text
    #     text = ""
    #     for cand_page in candidates:
    #         entity_id = cand_page["entity_id"]
    #         canonical_name = cand_page["canonical_name"]
    #         # desc = cand_page["description"]
    #         text += f"* {entity_id}: {canonical_name}\n"
    #     return text.rstrip()

    def generate_input_mention_candidates_pairs_prompt(
        self,
        document,
        candidate_entity_pages_for_doc,
        target_mention_indices,
        demonstration_mode=False
    ):
        # Aggregate mentions names
        words = " ".join(document["sentences"]).split()
        names = []
        for m_i, mention in enumerate(document["mentions"]):
            if m_i in target_mention_indices:
                begin_i, end_i = mention["span"]
                name = " ".join(words[begin_i: end_i + 1])
                names.append(name)

        # Aggregate candidate concepts for each mention
        cands_list = []
        for m_i, candidate_entity_pages_for_one_mention in (
            enumerate(candidate_entity_pages_for_doc)
        ):
            if m_i in target_mention_indices:
                cands = [] 
                if demonstration_mode:
                    # Place the ground-truth entity at the end of the candidates
                    gold_entity_id = document["mentions"][m_i]["entity_id"]
                    gold_entity_page = self.entity_dict[gold_entity_id]
                    for cand_page in candidate_entity_pages_for_one_mention[:N_CAND]:
                        if cand_page["entity_id"] != gold_entity_id:
                            cands.append(cand_page)
                    cands = cands[:2] + [gold_entity_page]
                else:
                    for cand_page in candidate_entity_pages_for_one_mention[:N_CAND]:
                        cands.append(cand_page)
                cands_list.append(cands)

        assert len(target_mention_indices) == len(names) == len(cands_list)

        # Texturize
        text = ""
        for m_i, (name, cands) in enumerate(zip(names, cands_list)):
            text += f"Mention {m_i + 1}: {name}\n"
            text += f"Candidate Concept IDs for Mention {m_i + 1}:\n"
            for cand_page in cands:
                entity_id = cand_page["entity_id"].replace("|", " ")
                canonical_name = cand_page["canonical_name"].replace("|", " ")
                desc = cand_page["description"].replace("|", " ").replace("\n", " ").rstrip()
                text += f"- ID: {entity_id} | Name: {canonical_name} | Description: {desc}\n"
                # text += f"   - {entity_id}: {canonical_name}\n"
        return text.rstrip()

    def generate_output_prompt(self, document, candidate_entity_pages_for_doc, target_mention_indices):
        """
        Parameters
        ----------
        document : Document
        candidate_entity_pages_for_doc: list[list[EntityPage]]
        target_mention_indices : list[int]

        Returns
        -------
        str
        """
        # text = ""
        # words = " ".join(document["sentences"]).split()
        # for m_i, mention in enumerate(document["mentions"]):
        #     if m_i in target_mention_indices:
        #         begin_i, end_i = mention["span"]
        #         name = " ".join(words[begin_i : end_i + 1])
        #         entity_id = mention["entity_id"]
        #         text += f"- {name} -> {entity_id}\n"
        # return text.rstrip()

        output_json = {
            "entity_disambiguation": []
        }

        words = " ".join(document["sentences"]).split()
        for m_i, mention in enumerate(document["mentions"]):
            if m_i in target_mention_indices:
                begin_i, end_i = mention["span"]
                name = " ".join(words[begin_i : end_i + 1])

                entity_id = mention["entity_id"]

                # If the ground-truth entity cannot be found in the candidates, set the target output "NA"
                if not entity_id in [epage["entity_id"] for epage in candidate_entity_pages_for_doc[m_i]]:
                    entity_id = "NA"

                output_json["entity_disambiguation"].append(
                    {
                        "mention": name,
                        "concept_id": entity_id
                    }
                )

        text = ""
        for m in output_json["entity_disambiguation"]:
            name = m["mention"]
            eid = m["concept_id"]
            text += f"- {name} | {eid}\n"
        return text.rstrip()

 
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
        extractor,
        documents,
        candidate_entities,
        split
    ):
        """
        Parameters
        ----------
        extractor : LLMED
        documents : list[Document]
        candidate_entities : list[dict[str, str | list[list[CandEntKeyInfo]]]]
        split : str
        """
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(extractor.prompt_processor.entity_dict.keys()))
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

    def save_extractor(self, extractor):
        """
        Parameters
        ----------
        extractor : LLMED
        """
        # Since we do not finetune the model, we save the configuration
        #   by this function.
        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], extractor.config)
        utils.write_json(self.paths["path_config"], extractor.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

    def evaluate(
        self,
        extractor,
        documents,
        candidate_entities,
        demonstrations,
        contexts,
        split,
        #
        get_scores_only=False
    ):
        """
        Parameters
        ----------
        extractor : LLMED
        documents : list[Document]
        candidate_entities : list[dict[str, str | list[list[CandEntKeyInfo]]]]
        demonstrations : list[dict[str, str | list[DemoKeyInfo]]]
        contexts : list[dict[str, str | list[Passage]]] | None
        split : str
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # (documents, candidate_entities, demonstrations, contexts) -> path_pred
        result_documents = extractor.batch_extract(
            documents=documents,
            candidate_entities=candidate_entities,
            demonstrations=demonstrations,
            contexts=contexts
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
