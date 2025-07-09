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
from ..demonstration_retrieval import DemonstrationRetriever
from .. import utils
from ..datatypes import (
    Config,
    Document,
    CandidateEntitiesForDocument,
    DemonstrationsForOneExample
)


logger = logging.getLogger(__name__)


def load(
    identifier: str,
    gpu_map: dict[str,int] | None = None,
) -> Pipeline:
    return Pipeline(identifier=identifier, gpu_map=gpu_map)


class Pipeline:

    def __init__(
        self,
        identifier: str,
        gpu_map: dict[str,int] | None = None,
    ):
        self.identifier = identifier
        self.gpu_map = gpu_map or {
            "ner": 0, "ed_retrieval": 0, "ed_reranking": 0, "docre": 0
        }

        # Load the pipeline Config 
        self.root_config: Config = utils.get_hocon_config(
            os.path.join(expanduser("~"), ".kapipe", "config")
        )
        self.pipe_config: Config = self.root_config[self.identifier]

        # Initialize the Chunker
        self.chunker = Chunker()

        # Initialize the NER wrapper
        self.ner = NER(
            task_config=self.pipe_config["ner"],
            gpu=self.gpu_map["ner"],
        )
        llm_model = (
            getattr(self.ner.extractor, "model", None)
            if self.pipe_config["ner"]["task_method"] == "llmner" else None
        )

        # Initialize the ED-Retrieval wrapper
        self.ed_ret = EDRetrieval(
            task_config=self.pipe_config["ed_retrieval"],
            gpu=self.gpu_map["ed_retrieval"],
        )

        # Initialize the ED-Reranking wrapper
        self.ed_rank = EDReranking(
            task_config=self.pipe_config["ed_reranking"],
            gpu=self.gpu_map["ed_reranking"],
            llm_model=llm_model,
        )
        llm_model = (
            getattr(self.ed_rank.extractor, "model", None)
            if self.pipe_config["ed_reranking"]["task_method"] == "llmed" else None
        )

        # Initialize the DocRE wrapper
        self.docre = DocRE(
            task_config=self.pipe_config["docre"],
            gpu=self.gpu_map["docre"],
            llm_model=llm_model,
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

    def __init__(self, task_config: Config, gpu: int = 0):
        self.task_config = task_config
        self.gpu = gpu

        # Initialize the NER extractor
        if self.task_config["task_method"] == "biaffinener":
            self.extractor = BiaffineNER(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.task_config["snapshot"]
            )
        elif self.task_config["task_method"] == "llmner":
            self.extractor = LLMNER(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.task_config["snapshot"],
                model=None,
            )
            self.demonstration_retriever = DemonstrationRetriever(
                path_demonstration_pool=self.extractor.path_demonstration_pool,
                method="count",
                task="ner"
            )
        else:
            raise Exception(f"Invalid task_method: {self.task_config['task_method']}")

    def __call__(self, document: Document) -> Document:
        if self.task_config["task_method"] == "llmner":
            # Get demonstrations for this document
            demonstrations_for_doc: DemonstrationsForOneExample = (
                self.demonstration_retriever.search(
                    document=document,
                    top_k=5
                )
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
    
    def __init__(self, task_config: Config, gpu : int = 0):
        self.task_config = task_config
        self.gpu = gpu
       
        # Initialize the ED-Retrieval extractor 
        if self.task_config["task_method"] == "blink": 
            self.extractor = BlinkBiEncoder(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.task_config["snapshot"]
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
    
    def __init__(self, task_config: Config, gpu: int = 0, llm_model: Any = None):
        self.task_config = task_config
        self.gpu = gpu
       
        # Initialize the ED-Reranking extractor 
        if self.task_config["task_method"] == "none":
            self.extractor = None
        elif self.task_config["task_method"] == "blink":
            self.extractor = BlinkCrossEncoder(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.task_config["snapshot"]
            )
        elif self.task_config["task_method"] == "llmed":
            self.extractor = LLMED(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.task_config["snapshot"],
                model=llm_model,
            )
            # Initialize the demonstration retriever
            self.demonstration_retriever = DemonstrationRetriever(
                path_demonstration_pool=self.extractor.path_demonstration_pool,
                method="count",
                task="ed"
            )
        else:
            raise Exception(f"Invalid task_method: {self.task_config['task_method']}")

    def __call__(
        self,
        document: Document,
        candidate_entities: CandidateEntitiesForDocument
    ) -> Document:
        # Skip the reranking
        if self.extractor is None:
            return document

        # Apply the extractor to the candidate entities
        return self.extractor.extract(
            document=document,
            candidate_entities_for_doc=candidate_entities
        )


class DocRE:

    def __init__(self, task_config: Config, gpu: int = 0, llm_model: Any = None):
        self.task_config = task_config
        self.gpu = gpu

        # Initialize the DocRE extractor
        if self.task_config["task_method"] == "atlop":
            self.extractor = ATLOP(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.task_config["snapshot"]
            )
        elif self.task_config["task_method"] == "llmdocre":
            self.extractor = LLMDocRE(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.task_config["snapshot"],
                model=llm_model,
            )
            # Initialize the demonstration retriever
            self.demonstration_retriever = DemonstrationRetriever(
                path_demonstration_pool=self.extractor.path_demonstration_pool,
                method="count",
                task="docre"
            )
        else:
            raise Exception(f"Invalid task_method: {self.task_config['task_method']}")

    def __call__(self, document: Document) -> Document:
        if self.task_config["task_method"] == "llmdocre":
            # Get demonstrations for this document
            demonstrations_for_doc: DemonstrationsForOneExample = (
                self.demonstration_retriever.search(
                    document=document,
                    top_k=5
                )
            )
            # Apply the extractor to the document
            return self.extractor.extract(
                document=document,
                demonstrations_for_doc=demonstrations_for_doc
            )
        else:
            # Apply the extractor to the document
            return self.extractor.extract(document=document)
