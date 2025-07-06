from __future__ import annotations

import logging
import os
from os.path import expanduser

from typing import Any

from ..chunking import Chunker
from . import BiaffineNER, LLMNER
from . import BlinkBiEncoder
from . import BlinkCrossEncoder, LLMED
from . import ATLOP, LLMDocRE
from ..demonstration_retrievers import DemonstrationRetriever
from .. import utils
from ..datatypes import (
    Config,
    Passage,
    Document,
    CandidateEntitiesForDocument,
    DemonstrationsForOneExample
)


logger = logging.getLogger(__name__)


def load(
    identifier: str,
    gpu_map: dict[str,int] | None = None,
    verbose: bool = True
) -> Pipeline:
    return Pipeline(identifier=identifier, gpu_map=gpu_map, verbose=verbose)


class Pipeline:

    def __init__(
        self,
        identifier: str,
        gpu_map: dict[str,int] | None = None,
        verbose: bool = True
    ):
        self.identifier = identifier
        self.gpu_map = gpu_map or {"ner": 0, "ed_retrieval": 0, "ed_reranking": 0, "docre": 0}

        # Load the pipeline Config 
        self.root_config: Config = utils.get_hocon_config(os.path.join(expanduser("~"), ".kapipe", "config"))
        self.pipe_config: Config = self.root_config[self.identifier]

        # Initialize the Chunker
        self.chunker = Chunker()

        # Initialize the NER wrapper
        self.ner = NER(
            task_config=self.pipe_config["ner"],
            gpu=self.gpu_map["ner"],
            verbose=verbose
        )
        llm_model = getattr(self.ner.extractor, "model", None) if self.pipe_config["ner"]["task_method"] == "llmner" else None

        # Initialize the ED-Retrieval wrapper
        self.ed_ret = EDRetrieval(
            task_config=self.pipe_config["ed_retrieval"],
            gpu=self.gpu_map["ed_retrieval"],
            verbose=verbose
        )

        # Initialize the ED-Reranking wrapper
        self.ed_rank = EDReranking(
            task_config=self.pipe_config["ed_reranking"],
            gpu=self.gpu_map["ed_reranking"],
            llm_model=llm_model,
            verbose=verbose
        )
        llm_model = getattr(self.ed_rank.extractor, "model", None) if self.pipe_config["ed_reranking"]["task_method"] == "llmed" else None

        # Initialize the DocRE wrapper
        self.docre = DocRE(
            task_config=self.pipe_config["docre"],
            gpu=self.gpu_map["docre"],
            llm_model=llm_model,
            verbose=verbose
        )

    def text_to_document(
        self,
        doc_key: str,
        text: str,
        title: str | None = None
    ) -> Document:
        # Split the text to (tokenized) sentences
        sentences = self.chunker.split_text_to_tokenized_sentences(text=text)
        sentences = [" ".join(s) for s in sentences]
        # Prepend the title as the first (tokenized) sentence
        if title:
            title = self.chunker.split_text_to_tokens(text=title)
            title = " ".join(title)
            sentences = [title] + sentences
        # Clean up the sentences
        sentences = self.chunker.remove_line_breaks(sentences=sentences)
        # Create a Document object
        document = {
            "doc_key": doc_key,
            "source_text": text,
            "sentences": sentences
        }
        return document
       
    def __call__(self, document: Document, num_candidate_entities: int = 10) -> Document:
        # Apply the NER wrapper
        document = self.ner(document=document)
        # Apply the ED-Retrieval wrapper
        document, candidate_entities = self.ed_ret(
            document=document,
            num_candidate_entities=num_candidate_entities
        )
        # Apply the ED-Reranking wrapper
        document = self.ed_rank(
            document=document,
            candidate_entities=candidate_entities
        )
        # Apply the DocRE wrapper
        document = self.docre(document=document)
        return document


class NER:

    def __init__(self, task_config: Config, gpu: int = 0, verbose: bool = True):
        self.task_config = task_config
        self.gpu = gpu
        self.verbose= verbose

        # Initialize the NER extractor
        if self.task_config["task_method"] == "biaffinener":
            device = f"cuda:{self.gpu}"
            path_config = os.path.join(self.task_config["basepath"], "config")
            path_vocab_etype = os.path.join(self.task_config["basepath"], "entity_types.vocab.txt")
            path_model = os.path.join(self.task_config["basepath"], "model")
            self.extractor = BiaffineNER(
                # General
                device=device,
                config=path_config,
                # Task specific
                vocab_etype=path_vocab_etype,
                # Misc.
                path_model=path_model,
                verbose=verbose
            )
        elif self.task_config["task_method"] == "llmner":
            device = f"cuda:{self.gpu}"
            path_config = os.path.join(self.task_config["basepath"], "config")
            path_vocab_etype = os.path.join(self.task_config["basepath"], "entity_types.vocab.txt")
            path_etype_meta_info = os.path.join(self.task_config["basepath"], "etype_meta_info.json")
            path_demonstration_pool = os.path.join(self.task_config["basepath"], "demonstration_pool.json")
            self.extractor = LLMNER(
                # General
                device=device,
                config=path_config,
                # Task specific
                vocab_etype=path_vocab_etype,
                etype_meta_info=path_etype_meta_info,
                # Optional
                path_demonstration_pool=path_demonstration_pool,
                # Misc.
                model=None,
                verbose=verbose
            )
            self.demonstration_retriever = DemonstrationRetriever(
                path_demonstration_pool=path_demonstration_pool,
                method="count",
                task="ner"
            )
        else:
            raise Exception(f"Invalid task_method: {self.task_config['task_method']}")

    def __call__(self, document: Document) -> Document:
        if self.task_config["task_method"] == "llmner":
            # Get demonstrations for this document
            demonstrations_for_doc: DemonstrationsForOneExample = self.demonstration_retriever.search(
                document=document,
                top_k=5
            )
            # Apply the extractor to the document
            return self.extractor.extract(
                document=document,
                demonstrations_for_doc=demonstrations_for_doc
            )
        else:
            # Apply the extractor to the document
            return self.extractor.extract(document=document)


class EDRetrieval:
    
    def __init__(self, task_config: Config, gpu : int = 0, verbose: bool = True):
        self.task_config = task_config
        self.gpu = gpu
        self.verbose = verbose
       
        # Initialize the ED-Retrieval extractor 
        if self.task_config["task_method"] == "blink": 
            device = f"cuda:{self.gpu}"
            path_config = os.path.join(self.task_config["basepath"], "config")
            path_model = os.path.join(self.task_config["basepath"], "model")
            path_entity_dict = os.path.join(self.task_config["basepath"], "entity_dict.json")
            self.extractor = BlinkBiEncoder(
                # General
                device=device,
                config=path_config,
                # Task specific
                path_entity_dict=path_entity_dict,
                # Misc.
                path_model=path_model,
                verbose=verbose
            )
            # Build the index based on the pre-computed embeddings
            self.extractor.make_index(use_precomputed_entity_vectors=True)
        else:
            raise Exception(f"Invalid task_method: {self.task_config['task_method']}")

    def __call__(
        self,
        document: Document,
        num_candidate_entities: int = 10
    ) -> tuple[Document, CandidateEntitiesForDocument]:
        # Apply the extractor to the document
        return self.extractor.extract(
            document=document,
            retrieval_size=num_candidate_entities
        )

 
class EDReranking:
    
    def __init__(self, task_config: Config, gpu: int = 0, llm_model: Any = None, verbose: bool = True):
        self.task_config = task_config
        self.gpu = gpu
        self.verbose = verbose
       
        # Initialize the ED-Reranking extractor 
        if self.task_config["task_method"] == "none":
            self.extractor = None
        elif self.task_config["task_method"] == "blink":
            device = f"cuda:{self.gpu}"
            path_config = os.path.join(self.task_config["basepath"], "config")
            path_model = os.path.join(self.task_config["basepath"], "model")
            path_entity_dict = os.path.join(self.task_config["basepath"], "entity_dict.json")
            self.extractor = BlinkCrossEncoder(
                # General
                device=device,
                config=path_config,
                # Task specific
                path_entity_dict=path_entity_dict,
                # Misc.
                path_model=path_model,
                verbose=verbose
            )
        elif self.task_config["task_method"] == "llmed":
            device = f"cuda:{self.gpu}"
            path_config = os.path.join(self.task_config["basepath"], "config")
            path_entity_dict = os.path.join(self.task_config["basepath"], "entity_dict.json")
            path_demonstration_pool = os.path.join(self.task_config["basepath"], "demonstration_pool.json")
            path_candidate_entities_pool = os.path.join(self.task_config["basepath"], "candidate_entities_pool.json")
            self.extractor = LLMED(
                # General
                device=device,
                config=path_config,
                # Task specific
                path_entity_dict=path_entity_dict,
                # Optional
                path_demonstration_pool=path_demonstration_pool,
                path_candidate_entities_pool=path_candidate_entities_pool,
                # Misc.
                model=llm_model,
                verbose=verbose
            )
            # Initialize the demonstration retriever
            self.demonstration_retriever = DemonstrationRetriever(
                path_demonstration_pool=path_demonstration_pool,
                method="count",
                task="ed"
            )
        else:
            raise Exception(f"Invalid task_method: {self.task_config['task_method']}")

    def __call__(self, document: Document, candidate_entities: CandidateEntitiesForDocument) -> Document:
        # Skip the reranking
        if self.extractor is None:
            return document

        # Apply the extractor to the candidate entities
        return self.extractor.extract(
            document=document,
            candidate_entities_for_doc=candidate_entities
        )


class DocRE:

    def __init__(self, task_config: Config, gpu: int = 0, llm_model: Any = None, verbose: bool = True):
        self.task_config = task_config
        self.gpu = gpu
        self.verbose = verbose

        # Initialize the DocRE extractor
        if self.task_config["task_method"] == "atlop":
            device = f"cuda:{self.gpu}"
            path_config = os.path.join(self.task_config["basepath"], "config")
            path_vocab_relation = os.path.join(self.task_config["basepath"], "relations.vocab.txt")
            path_model = os.path.join(self.task_config["basepath"], "model")
            self.extractor = ATLOP(
                # General
                device=device,
                config=path_config,
                # Task specific
                vocab_relation=path_vocab_relation,
                # Misc.
                path_model=path_model,
                verbose=verbose
            )
        elif self.task_config["task_method"] == "llmdocre":
            device = f"cuda:{self.gpu}"
            path_config = os.path.join(self.task_config["basepath"], "config")
            path_vocab_relation = os.path.join(self.task_config["basepath"], "relations.vocab.txt")
            path_rel_meta_info = os.path.join(self.task_config["basepath"], "rel_meta_info.json")
            path_entity_dict = os.path.join(self.task_config["basepath"], "entity_dict.json")
            path_demonstration_pool = os.path.join(self.task_config["basepath"], "demonstration_pool.json")
            self.extractor = LLMDocRE(
                # General
                device=device,
                config=path_config,
                # Task specific
                vocab_relation=path_vocab_relation,
                rel_meta_info=path_rel_meta_info,
                # Method specific
                path_entity_dict=path_entity_dict,
                # Optional: few-shot setting
                path_demonstration_pool=path_demonstration_pool,
                # Misc.
                model=llm_model,
                verbose=verbose
            )
            # Initialize the demonstration retriever
            self.demonstration_retriever = DemonstrationRetriever(
                path_demonstration_pool=path_demonstration_pool,
                method="count",
                task="docre"
            )
        else:
            raise Exception(f"Invalid task_method: {self.task_config['task_method']}")

    def __call__(self, document: Document) -> Document:
        if self.task_config["task_method"] == "llmdocre":
            # Get demonstrations for this document
            demonstrations_for_doc: DemonstrationsForOneExample = self.demonstration_retriever.search(
                document=document,
                top_k=5
            )
            # Apply the extractor to the document
            return self.extractor.extract(
                document=document,
                demonstrations_for_doc=demonstrations_for_doc
            )
        else:
            # Apply the extractor to the document
            return self.extractor.extract(document=document)
