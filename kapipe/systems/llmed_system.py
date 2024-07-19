import copy
import logging

# import numpy as np
import torch
from tqdm import tqdm

from ..prompt_processors import EDPromptProcessorV5
from ..models import LLM
from .. import utils


logger = logging.getLogger(__name__)


class LLMEDSystem:

    def __init__(
        self,
        device,
        config,
        path_entity_dict,
        path_demonstration_pool,
        path_candidate_entities_pool,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        path_entity_dict: str
        path_demonstration_pool: str
        path_candidate_entities_pool: str
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> LLMEDSystem Initialization >>>>>>>>>>")
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

        self.n_demonstrations = config["n_demonstrations"]

        # dict[DocKey, Document]
        self.demonstration_pool = {
            demo_doc["doc_key"]: demo_doc
            for demo_doc in utils.read_json(path_demonstration_pool)
        }
        if self.verbose:
            logger.info(f"Loaded demonstration pool from {path_demonstration_pool}")

        # dict[str, EntityPage]
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        if self.verbose:
            logger.info(f"Loaded entity dictionary from {path_entity_dict}")

        # dict[DocKey, dict[str, list[list[CandEntKeyInfo]]]]
        self.candidate_entities_pool = {
            cands["doc_key"]: cands
            for cands in utils.read_json(path_candidate_entities_pool)
        }
        if self.verbose:
            logger.info(f"Loaded candidate entities for demonstration pool from {path_candidate_entities_pool}")

        if config["prompt_template_name_or_path"] == "ed_04":
            self.prompt_processor = EDPromptProcessorV5(
                prompt_template_name_or_path=\
                    config["prompt_template_name_or_path"],
                knowledge_base_name_prompt=config["knowledge_base_name"]
            )
        else:
            raise Exception(f"Invalid prompt_template_name_or_path: {config['prompt_template_name_or_path']}")

        ######
        # Model
        ######

        self.model_name = config["model_name"]
        assert self.model_name == "llm"

        self.model = LLM(
            device=device,
            llm_name_or_path=config["llm_name_or_path"],
            max_seg_len=config["max_seg_len"],
            max_new_tokens=config["max_new_tokens"],
            beam_size=config["beam_size"],
            do_sample=config["do_sample"],
            num_return_sequences=config["num_return_sequences"],
            clean_up_tokenization_spaces=config["clean_up_tokenization_spaces"],
            stop_list=config["stop_list"],
            quantization_bits=config["quantization_bits"]
        )
        # self.model.llm.to(self.model.device)

        if self.verbose:
            logger.info("<<<<<<<<<< LLMEDSystem Initialization <<<<<<<<<<")

    def extract(
        self,
        document,
        candidate_entities_for_doc,
        demonstrations_for_doc,
        return_multiple_outputs=False
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, list[list[CandEntKeyInfo]]]
        demonstrations_for_doc : dict[str, list[DemoKeyInfo]]
        return_multiple_outputs : bool
            by default False

        Returns
        -------
        Document | list[Document]
        """
        with torch.no_grad():
            # Switch to inference mode
            self.model.llm.eval()

            # Generate a prompt
            # Prepare candidate entities for the input document
            candidate_entity_dicts_for_doc = [] # list[list[EntityPage]]
            for candidate_entities_for_one_mention in (
                candidate_entities_for_doc["candidate_entities"]
            ):
                candidate_entity_dicts_for_one_mention = [
                    self.entity_dict[cand_dict["entity_id"]]
                    for cand_dict in candidate_entities_for_one_mention
                ]
                candidate_entity_dicts_for_doc.append(
                    candidate_entity_dicts_for_one_mention
                )
            # Prepare demonstration documents and candidate entities
            #   for the demonstration documents.
            demonstration_documents = [] # list[Document]
            candidate_entity_dicts_for_demos = [] # list[list[list[EntityPage]]]
            for demo_dict in (
                demonstrations_for_doc["demonstrations"][:self.n_demonstrations]
            ):
                # Demonstration document
                demo_doc = self.demonstration_pool[demo_dict["doc_key"]]
                demonstration_documents.append(demo_doc)

                # Candidate entities for the demonstration document
                # dict[str, list[list[CandEntKeyInfo]]]
                candidate_entities_for_demo \
                    = self.candidate_entities_pool[demo_dict["doc_key"]]
                # list[list[EntityPage]]
                candidate_entity_dicts_for_demo = []
                for candidate_entities_for_one_mention in (
                    candidate_entities_for_demo["candidate_entities"]
                ):
                    # list[EntityPage]
                    candidate_entity_dicts_for_one_mention = [
                        self.entity_dict[cand_dict["entity_id"]]
                        for cand_dict in candidate_entities_for_one_mention
                    ]
                    candidate_entity_dicts_for_demo.append(
                        candidate_entity_dicts_for_one_mention
                    )
                candidate_entity_dicts_for_demos.append(
                    candidate_entity_dicts_for_demo
                )
            prompt = self.prompt_processor.encode(
                document=document,
                candidate_entity_dicts_for_doc=candidate_entity_dicts_for_doc,
                demonstration_documents=demonstration_documents,
                candidate_entity_dicts_for_demos=\
                    candidate_entity_dicts_for_demos
            )

            # Preprocess
            preprocessed_data = self.model.preprocess(prompt=prompt)

            if preprocessed_data["skip"]:
                logger.info(f"Skipped generation because the text is too long ({len(preprocessed_data['llm_input']['segments_id'][0])}).")
                generated_texts = ["SKIPPED BECAUSE THE TEXT IS TOO LONG"]
            else:
                # Tensorize
                model_input = self.model.tensorize(
                    preprocessed_data=preprocessed_data,
                    compute_loss=False
                )

                # Forward
                generated_texts = self.model.generate(**model_input) # list[str]
                generated_texts = [
                    self.model.remove_prompt_from_generated_text(
                        generated_text=generated_text
                    )
                    for generated_text in generated_texts
                ]

            # Structurize
            mentions_list = [] # list[list[Mention]]
            entities_list = [] # list[list[Entity]]
            for generated_text in generated_texts:
                mentions, entities = self.structurize(
                    document=document,
                    generated_text=generated_text
                )
                mentions_list.append(mentions)
                entities_list.append(entities)

            # Integrate
            result_documents = [] # list[Document]
            for mentions, entities, generated_text in zip(
                mentions_list,
                entities_list,
                generated_texts
            ):
                result_document = copy.deepcopy(document)
                for m_i in range(len(result_document["mentions"])):
                    result_document["mentions"][m_i].update(mentions[m_i])
                result_document["entities"] = entities
                result_document["ed_prompt"] = preprocessed_data["prompt"]
                result_document["ed_generated_text"] = generated_text
                result_documents.append(result_document)
            if return_multiple_outputs:
                return result_documents
            else:
                return result_documents[0]

    def structurize(self, document, generated_text):
        # Transform to mention-level entity IDs
        mentions = self.prompt_processor.decode(
            generated_text=generated_text,
            document=document
        )

        # Transform to entity-level entity IDs
        # i.e., aggregate mentions based on the entity IDs
        entity_id_to_info = {} # dict[str, dict[str, Any]]
        for m_i in range(len(document["mentions"])):
            entity_type = document["mentions"][m_i]["entity_type"]
            entity_id = mentions[m_i]["entity_id"]
            # gen_line = mentions[m_i]["corresponding_output_line"]
            if entity_id in entity_id_to_info:
                entity_id_to_info[entity_id]["mention_indices"].append(m_i)
                # entity_id_to_info[entity_id][
                #     "corresponding_output_lines_by_ed"
                # ].append(gen_line)
                # TODO
                # Confliction of entity types can appear, if EL model does not care about it.
                # assert entity_id_to_info[entity_id]["entity_type"] == entity_type
            else:
                entity_id_to_info[entity_id] = {}
                entity_id_to_info[entity_id]["mention_indices"] = [m_i]
                # TODO
                entity_id_to_info[entity_id]["entity_type"] = entity_type
                # entity_id_to_info[entity_id][
                #     "corresponding_output_lines_by_ed"
                # ] = [gen_line]

        entities = [] # list[Entity]
        for entity_id in entity_id_to_info.keys():
            mention_indices \
                = entity_id_to_info[entity_id]["mention_indices"]
            entity_type = entity_id_to_info[entity_id]["entity_type"]
            # gen_lines = entity_id_to_info[entity_id][
            #     "corresponding_output_lines_by_ed"
            # ]
            entities.append({
                "mention_indices": mention_indices,
                "entity_type": entity_type,
                "entity_id": entity_id,
                # "corresponding_output_lines_by_ed": gen_lines,
            })

        return mentions, entities

    def batch_extract(self, documents, candidate_entities, demonstrations):
        """
        Parameters
        ----------
        documents : list[Document]
        candidate_entities: list[dict[str, list[list[CandEntKeyInfo]]]]
        demonstrations: list[dict[str, list[DemoKeyInfo]]]

        Returns
        -------
        list[Document]
        """
        result_documents = []
        for document, candidate_entities_for_doc, demonstrations_for_doc in \
            tqdm(
                zip(documents, candidate_entities, demonstrations),
                total=len(documents),
                desc="extraction steps"
            ):
            result_document = self.extract(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc,
                demonstrations_for_doc=demonstrations_for_doc
            )
            result_documents.append(result_document)
        return result_documents

