import copy
import logging

# import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm

from ..prompt_processors import DocREPromptProcessorV4
from ..models import LLM
from .. import utils


logger = logging.getLogger(__name__)


class LLMDocRESystem:

    def __init__(
        self,
        device,
        config,
        path_entity_dict,
        vocab_relation,
        path_demonstration_pool,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        path_entity_dict: str
        vocab_relation: dict[str, int] | str
        path_demonstration_pool: str
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> LLMDocRESystem Initialization >>>>>>>>>>")
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

        if config["prompt_template_name_or_path"] == "docre_03":
            self.prompt_processor = DocREPromptProcessorV4(
                prompt_template_name_or_path=\
                    config["prompt_template_name_or_path"],
                entity_dict=self.entity_dict,
                #
                rel_name_to_pretty_rel_name=\
                    config["rel_name_to_pretty_rel_name"],
                knowledge_base_name_prompt=config["knowledge_base_name"],
                mention_style=config["mention_style"],
                with_span_annotation=config["with_span_annotation"]
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
            logger.info("<<<<<<<<<< LLMDocRESystem Initialization <<<<<<<<<<")

    def extract(
        self,
        document,
        demonstrations_for_doc,
        return_multiple_outputs=False
    ):
        """
        Parameters
        ----------
        document : Document
        demonstrations_for_doc : dict[str, list[DemoKeyInfo]]
        return_multiple_outputs : bool
            by default False

        Returns
        -------
        Document | list[Document]
        """
        if len(document["entities"]) <= 1:
            result_document = copy.deepcopy(document)
            result_document["relations"] = []
            result_document["docre_prompt"] = ""
            result_document["docre_generated_text"] = ""
            if return_multiple_outputs:
                return [result_document]
            else:
                return result_document

        with torch.no_grad():
            # Switch to inference mode
            self.model.llm.eval()

            # Generate a prompt
            demo_docs = [] # list[Document]
            demo_dicts = demonstrations_for_doc["demonstrations"]
            for demo_dict in demo_dicts[:self.n_demonstrations]:
                demo_doc = self.demonstration_pool[demo_dict["doc_key"]]
                demo_docs.append(demo_doc)
            prompt = self.prompt_processor.encode(
                document=document,
                demonstration_documents=demo_docs
            )
    
            # Preprocesss
            preprocessed_data = self.model.preprocess(prompt=prompt)

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
            triples_list = [] # list[list[Triple]]
            for generated_text in generated_texts:
                triples = self.prompt_processor.decode(
                    generated_text=generated_text,
                    document=document
                )
                triples_list.append(triples)

            # Integrate
            result_documents = [] # list[Document]
            for triples, generated_text in zip(triples_list, generated_texts):
                result_document = copy.deepcopy(document)
                result_document["relations"] = triples
                result_document["docre_prompt"] = preprocessed_data["prompt"]
                result_document["docre_generated_text"] = generated_text
                result_documents.append(result_document)
            if return_multiple_outputs:
                return result_documents
            else:
                return result_documents[0]

    def batch_extract(self, documents, demonstrations):
        """
        Parameters
        ----------
        documents : list[Document]
        demonstrations : list[dict[str, list[DemoKeyInfo]]]

        Returns
        -------
        list[Document]
        """
        result_documents = []
        for document, demonstrations_for_doc in tqdm(
            zip(documents, demonstrations),
            total=len(documents),
            desc="extraction steps"
        ):
            result_document = self.extract(
                document=document,
                demonstrations_for_doc=demonstrations_for_doc
            )
            result_documents.append(result_document)
        return result_documents

