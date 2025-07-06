import copy
# import json
import logging
import re

# import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm

from ..models import LLM
from ..models import OpenAILLM
from .. import utils


logger = logging.getLogger(__name__)


class LLMDocRE:

    def __init__(
        self,
        # General
        device,
        config,
        # Task specific
        vocab_relation,
        rel_meta_info,
        # Method specific
        path_entity_dict,
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
        vocab_relation: dict[str, int] | str
        rel_meta_info: dict[str, dict[str, str]] | str
        path_entity_dict: str
        path_demonstration_pool: str | None
            by default None
        model : LLM | None
            by default None
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> LLMDocRE Initialization >>>>>>>>>>")
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
        # Vocabulary (relations)
        ######

        if isinstance(vocab_relation, str):
            tmp = vocab_relation
            vocab_relation = utils.read_vocab(vocab_relation)
            if self.verbose:
                logger.info(f"Loaded relation type vocabulary from {tmp}")
        self.vocab_relation = vocab_relation
        self.ivocab_relation = {i:l for l, i in self.vocab_relation.items()}

        if isinstance(rel_meta_info, str):
            tmp = rel_meta_info
            rel_meta_info = utils.read_json(rel_meta_info)
            if self.verbose:
                logger.info(f"Loaded relation meta-information from {tmp}")
        self.rel_meta_info = rel_meta_info

        ######
        # Prompt Processor
        ######

        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=\
                config["prompt_template_name_or_path"],
            path_entity_dict=path_entity_dict,
            vocab_relation=self.vocab_relation,
            rel_meta_info=self.rel_meta_info,
            knowledge_base_name_prompt=config["knowledge_base_name"],
            mention_style=config["mention_style"],
            path_demonstration_pool=path_demonstration_pool,
            n_demonstrations=config["n_demonstrations"] ,
            with_span_annotation=config["with_span_annotation"]
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
        # self.re_comp = re.compile(
        #     "(.+?)\s*\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)$"
        # )
        # <bullet> <head entity ID> -> <relation> -> <tail entity ID>
        # self.re_comp = re.compile(
        #     "(.+?)\s*(.+?)\s*->\s*(.+?)\s*->\s*(.+?)$"
        # )
        # <bullet> <head entity ID> | <relation> | <tail entity ID>
        self.re_comp = re.compile(
            "(.+?)\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$"
        )

        # e.g., "chemical-induce-disease" -> "CID"
        self.normalized_to_canonical = {}
        for rel in self.vocab_relation.keys():
            pretty_name = self.rel_meta_info[rel]["Pretty Name"]
            self.normalized_to_canonical[pretty_name.lower()] = rel

        if self.verbose:
            logger.info("<<<<<<<<<< LLMDocRE Initialization <<<<<<<<<<")

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
        if len(document["entities"]) <= 1:
            result_document = copy.deepcopy(document)
            result_document["relations"] = []
            result_document["docre_prompt"] = ""
            result_document["docre_generated_text"] = ""
            return result_document

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
                # Preprocesss
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
            # list[Triple]
            triples = self.structurize(
                document=document,
                generated_text=generated_text
            )

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["relations"] = triples
            if self.model_name == "llm":
                result_document["docre_prompt"] = preprocessed_data["prompt"]
            else:
                result_document["docre_prompt"] = prompt
            result_document["docre_generated_text"] = generated_text
            return result_document

    def structurize(self, document, generated_text):
        """
        Parameters
        ----------
        document : Document
        generated_text : str

        Returns
        -------
        list[Triple]
        """
        doc_key = document["doc_key"]

        # Get mapping from entity ID to entity index
        entity_id_to_index = {}
        for e_i, e in enumerate(document["entities"]):
            # entity_id_to_index[e["entity_id"]] = e_i
            entity_id_to_index[f"Entity{e_i}"] = e_i
            
        # Parse each generated line
        generated_lines = generated_text.split("\n")
        tuples = [] # list[tuple[int, str, int]]
        for generated_line in generated_lines:
            generated_line = generated_line.strip()
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

            # Check whether the relation can be found in the possible set
            normalized_relation = relation.lower() # e.g., "Chemical-Induce-Disease" -> "chemical-induce-relation"
            if not normalized_relation in self.normalized_to_canonical:
                logger.info(f"[{doc_key}] A generated line contains invalid relation: '{generated_line}'")
                # continue
            # canonical_relation \
            #     = self.normalized_to_canonical[normalized_relation] # e.g., "CID"
            canonical_relation = self.normalized_to_canonical.get(normalized_relation, relation)

            # Get entity index
            head_idx = entity_id_to_index[head_id]
            tail_idx = entity_id_to_index[tail_id]

            # Add new tuples
            tuple_ = (head_idx, canonical_relation, tail_idx)
            if not tuple_ in tuples:
                tuples.append(tuple_)

        # Convert tuples to triple dicts
        triples = [] # list[Triple]
        for tuple_ in tuples:
            arg1, rel, arg2 = tuple_
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

    # def structurize(self, document, generated_text):
    #     """
    #     Parameters
    #     ----------
    #     document : Document
    #     generated_text : str

    #     Returns
    #     -------
    #     list[Triple]
    #     """
    #     # Get mapping from entity ID to entity index
    #     entity_id_to_index = {}
    #     for e_i, e in enumerate(document["entities"]):
    #         entity_id_to_index[e["entity_id"]] = e_i

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
    #     if not "triples" in json_obj:
    #         logger.info(f"Skipped extraction because the parsed JSON object does not contain `triples' key: '{json_obj}'")
    #         return []

    #     # We process each entry in the list
    #     tuples = []
    #     for entry in json_obj["triples"]:
    #         if (not "subject" in entry) or (not "relation" in entry) or (not "object" in entry):
    #             logger.info(f"Skipped a parsed JSON entry of invalid formatting: '{entry}'")
    #             continue
    #         head_id = entry["subject"]
    #         relation = entry["relation"]
    #         tail_id = entry["object"]
            
    #         # Check whether the head/tail IDs can be found in the possible list
    #         if (
    #             (not head_id in entity_id_to_index)
    #             or
    #             (not tail_id in entity_id_to_index)
    #             or
    #             head_id == tail_id
    #         ):
    #             logger.info(f"Skipped a parsed JSON entry with invalid entity pair: '{entry}'")
    #             continue

    #         # Check whether the relation can be found in the possible set
    #         normalized_relation = relation.lower() # e.g., "Chemical-Induce-Disease" -> "chemical-induce-relation"
    #         if not normalized_relation in self.normalized_to_canonical:
    #             logger.info(f"Skipped a parsed JSON entry with invalid relation: '{entry}'")
    #             continue
    #         canonical_relation \
    #             = self.normalized_to_canonical[normalized_relation] # e.g., "CID"

    #         # Get entity index
    #         head_idx = entity_id_to_index[head_id]
    #         tail_idx = entity_id_to_index[tail_id]

    #         # Add new tuples
    #         tuple_ = (head_idx, canonical_relation, tail_idx)
    #         if not tuple_ in tuples:
    #             tuples.append(tuple_)

    #     # Convert tuples to triple dicts
    #     triples = [] # list[Triple]
    #     for tuple_ in tuples:
    #         arg1, rel, arg2 = tuple_
    #         triples.append({
    #             "arg1": arg1,
    #             "relation": rel,
    #             "arg2": arg2
    #         })

    #     triples = sorted(
    #         triples,
    #         key=lambda x: (x["arg1"], x["arg2"], x["relation"])
    #     )
    #     return triples

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
        path_entity_dict,
        vocab_relation,
        rel_meta_info,
        knowledge_base_name_prompt,
        mention_style,
        # optional: few-shot setting
        path_demonstration_pool=None,
        n_demonstrations=None,
        # misc.
        with_span_annotation=True
    ):
        """
        Parameters
        ----------
        prompt_template_name : str
        path_entity_dict : str
        vocab_relation: dict[str, int]
        rel_meta_info: dict[str, dict[str, str]]
        knowledge_base_name_prompt: str
        mention_style: str
        path_demonstration_pool: str | None
            by default None
        n_demonstrations: int | None
            by default None
        with_span_annotation : bool
            by default True
        """
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.path_entity_dict = path_entity_dict
        self.vocab_relation = vocab_relation
        self.rel_meta_info = rel_meta_info
        self.knowledge_base_name_prompt = knowledge_base_name_prompt
        self.mention_style = mention_style
        self.path_demonstration_pool = path_demonstration_pool
        self.n_demonstrations = n_demonstrations
        self.with_span_annotation = with_span_annotation

        assert self.mention_style in ["canonical_name", "first_mention", "all_mentions"]

        if self.path_demonstration_pool is not None:
            assert self.n_demonstrations is not None

        if self.n_demonstrations == 0:
            self.path_demonstration_pool = None

        #####
        # Load prompt template
        #####

        self.prompt_template = utils.read_prompt_template(prompt_template_name_or_path=self.prompt_template_name_or_path)

        # Check requirements
        assert "{relations_prompt}" in self.prompt_template # XXX
        if self.path_demonstration_pool is not None:
            assert "{demonstrations_prompt}" in self.prompt_template
        # assert "{contexts_prompt}" in self.prompt_template
        assert "{task_prompt}" in self.prompt_template

        self.relations_prompt = ""
        # for k, v in rel_name_to_pretty_rel_name.items():
        #     self.relations_prompt += f"- {v}\n"
        for rel in vocab_relation.keys():
            pretty_name = self.rel_meta_info[rel]["Pretty Name"]
            definition = self.rel_meta_info[rel]["Definition"]
            self.relations_prompt += f"- {pretty_name}: {definition}\n"
        self.relations_prompt = self.relations_prompt.rstrip()

       #####
        # Load entity dictionary
        #####

        # dict[str, EntityPage]
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }

        #####
        # Load demonstration pool
        #####

        if self.path_demonstration_pool is not None:
            # dict[DocKey, Document]
            self.demonstration_pool = {
                demo_doc["doc_key"]: demo_doc
                for demo_doc in utils.read_json(path_demonstration_pool)
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
            document=document,
        )

        # Combine the prompt parts
        prompt = self.prompt_template.format(
            knowledge_base_name_prompt=self.knowledge_base_name_prompt,
            relations_prompt=self.relations_prompt, # XXX
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
            text += (
                "Entities:\n"
                + self.generate_input_entities_prompt(document=demo_doc)
                + "\n"
            )
            # Output
            text += (
                "Output:\n"
                + self.generate_relations_prompt(
                    document=demo_doc
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
        text += (
            "Entities:\n"
            + self.generate_input_entities_prompt(document=document)
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

    def generate_input_entities_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        str
        """
        text = ""
        words = " ".join(document["sentences"]).split()
        mentions = document["mentions"]
        entities = document["entities"]
        for e_i, entity in enumerate(entities):
            entity_id = entity["entity_id"]
            entity_type = entity["entity_type"]
            if self.mention_style == "all_mentions":
                mention_indices = entity["mention_indices"]
                names = []
                for m_i in mention_indices:
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
                names = ", ".join([f"\"{n}\"" for n in names])
                # text += f"- {entity_id}: {names}\n"
                text += f"- Entity{e_i}: {names} ({entity_type})\n"
            elif self.mention_style == "first_mention":
                mention_indices = entity["mention_indices"]
                mention = mentions[mention_indices[0]]
                if self.with_span_annotation:
                    begin_i, end_i = mention["span"]
                    name = " ".join(words[begin_i: end_i + 1])
                else:
                    name = mention["name"]
                # text += f"- {entity_id}: \"{name}\"\n"
                text += f"- Entity{e_i}: \"{name}\" ({entity_type})\n"
            elif self.mention_style == "canonical_name":
                epage = self.entity_dict[entity_id]
                name = epage["canonical_name"]
                # text += f"- {entity_id}: {name}\n"
                text += f"- Entity{e_i}: {name} ({entity_type})\n"
            else:
                raise Exception(f"Invalid mention_style: {self.mention_style}")
        return text.rstrip()

    def generate_relations_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        str
        """
        # text = ""
        # # words = " ".join(document["sentences"]).split()
        # # pretty_rel_name = self.rel_name_to_pretty_rel_name[relation_type]
        # entities = document["entities"]
        # for triple in document["relations"]:
        #     head_idx = triple["arg1"]
        #     tail_idx = triple["arg2"]
        #     rel = triple["relation"]
        #     head_id = entities[head_idx]["entity_id"]
        #     tail_id = entities[tail_idx]["entity_id"]
        #     pretty_rel = self.rel_name_to_pretty_rel_name[rel]
        #     text += f"- {head_id} -> {pretty_rel} -> {tail_id}\n"
        # return text.rstrip()

        # output_json = {
        #     "triples": []
        # }
        # entities = document["entities"]
        # for triple in document["relations"]:
        #     head_idx = triple["arg1"]
        #     tail_idx = triple["arg2"]
        #     rel = triple["relation"]
        #     head_id = entities[head_idx]["entity_id"]
        #     tail_id = entities[tail_idx]["entity_id"]
        #     pretty_rel = self.rel_name_to_pretty_rel_name[rel]
        #     output_json["triples"].append(
        #         {
        #             "subject": head_id,
        #             "relation":  pretty_rel,
        #             "object": tail_id
        #         }
        #     )
        # # return json.dumps(output_json)
        # text = ""
        # for triple in output_json["triples"]:
        #     h = triple["subject"]
        #     r = triple["relation"]
        #     t = triple["object"]
        #     text += f"- {h} | {r} | {t}\n"
        # return text.rstrip()

        text = ""
        for triple in document["relations"]:
            head_idx = triple["arg1"]
            tail_idx = triple["arg2"]
            rel = triple["relation"]
            pretty_name = self.rel_meta_info[rel]["Pretty Name"]
            text += f"- Entity{head_idx} | {pretty_name} | Entity{tail_idx}\n"
        return text.rstrip()
 
 

