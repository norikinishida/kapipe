import copy
import logging

# import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm

from ..prompt_processors import NERPromptProcessorV5
from ..models import LLM
from .. import utils


logger = logging.getLogger(__name__)


class LLMNERSystem:

    def __init__(
        self,
        device,
        config,
        vocab_etype,
        path_demonstration_pool,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        vocab_etype: dict[str, int] | str
        path_demonstration_pool: str
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> LLMNERSystem Initialization >>>>>>>>>>")
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

        if config["prompt_template_name_or_path"] in ["ner_07"]:
            self.prompt_processor = NERPromptProcessorV5(
                prompt_template_name_or_path=config[
                    "prompt_template_name_or_path"
                ],
                possible_entity_types=list(self.vocab_etype.keys())
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
            logger.info("<<<<<<<<<< LLMNERSystem Initialization <<<<<<<<<<")

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
        with torch.no_grad():
            # Switch to inference mode
            self.model.llm.eval()

            # Generate a prompt
            demonstration_documents = [] # list[Document]
            for demo_dict in (
                demonstrations_for_doc["demonstrations"][:self.n_demonstrations]
            ):
                demo_doc = self.demonstration_pool[demo_dict["doc_key"]]
                demonstration_documents.append(demo_doc)
            prompt = self.prompt_processor.encode(
                document=document,
                demonstration_documents=demonstration_documents
            )
 
            # Preprocess
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
            ] # list[str]

            # Structurize
            mentions_list = [] # list[list[Mention]]
            for generated_text in generated_texts:
                mentions = self.prompt_processor.decode(
                    generated_text=generated_text,
                    document=document
                )
                mentions_list.append(mentions)

            # Integrate
            result_documents = [] # list[Document]
            for mentions, generated_text in zip(mentions_list, generated_texts):
                result_document = copy.deepcopy(document)
                result_document["mentions"] = mentions
                result_document["ner_prompt"] = preprocessed_data["prompt"]
                result_document["ner_generated_text"] = generated_text
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

