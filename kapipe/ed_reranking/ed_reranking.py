from __future__ import annotations

import logging
import os
from os.path import expanduser

from typing import Any

from .blinkcrossencoder import BlinkCrossEncoder
from .llmed import LLMED
from ..demonstration_retrieval import DemonstrationRetriever
from .. import utils
from ..datatypes import (
    Config,
    Document,
    CandidateEntitiesForDocument,
    DemonstrationsForOneExample
)


logger = logging.getLogger(__name__)


class EDReranking:
    
    def __init__(self, identifier: str, gpu: int = 0, llm_model: Any = None):
        self.identifier = identifier
        self.gpu = gpu

        root_config: Config = utils.get_hocon_config(
            os.path.join(expanduser("~"), ".kapipe", "download", "config")
        )
        self.module_config: Config = root_config["ed_reranking"][identifier]

        # # Download the configurations
        # utils.download_folder_if_needed(
        #     dest=self.module_config["snapshot"],
        #     url=self.module_config["url"]
        # )
 
        # Initialize the ED-Reranking reranker 
        if self.module_config["method"] == "none":
            self.reranker = None
        elif self.module_config["method"] == "blinkcrossencoder":
            self.reranker = BlinkCrossEncoder(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.module_config["snapshot"]
            )
        elif self.module_config["method"] == "llmed":
            self.reranker = LLMED(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.module_config["snapshot"],
                model=llm_model,
            )
            # Initialize the demonstration retriever
            self.demonstration_retriever = DemonstrationRetriever(
                path_demonstration_pool=self.reranker.prompt_processor.path_demonstration_pool,
                method="count",
                task="ed"
            )
        else:
            raise Exception(f"Invalid method: {self.module_config['method']}")

    def rerank(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument
    ) -> Document:
        # Skip the reranking
        if self.reranker is None:
            return document
        if self.module_config["method"] == "llmed":
            # Get demonstrations for this document
            demonstrations_for_doc: DemonstrationsForOneExample = (
                self.demonstration_retriever.search(
                    document=document,
                    top_k=5
                )
            )
            # Apply the reranker to the candidate entities
            return self.reranker.rerank(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc,
                demonstrations_for_doc=demonstrations_for_doc
            )
        else:
            # Apply the reranker to the candidate entities
            return self.reranker.rerank(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc
            )
            

