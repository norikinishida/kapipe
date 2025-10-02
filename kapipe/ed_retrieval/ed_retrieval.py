from __future__ import annotations

import logging
import os
from os.path import expanduser

from .dummy_entity_retriever import DummyEntityRetriever
from .blink_bi_encoder import BlinkBiEncoder
from .. import utils
from ..datatypes import (
    Config,
    Document,
    CandidateEntitiesForDocument,
)


logger = logging.getLogger(__name__)


class EDRetrieval:
    
    def __init__(self, identifier: str, gpu : int = 0):
        self.identifier = identifier
        self.gpu = gpu

        root_config: Config = utils.get_hocon_config(
            os.path.join(expanduser("~"), ".kapipe", "download", "config")
        )
        self.module_config: Config = root_config["ed_retrieval"][identifier]

        # # Download the configurations
        # utils.download_folder_if_needed(
        #     dest=self.module_config["snapshot"],
        #     url=self.module_config["url"]
        # )
       
        # Initialize the ED-Retrieval retriever
        if self.module_config["method"] == "dummy_entity_retriever":
            self.retriever = DummyEntityRetriever()
        elif self.module_config["method"] == "blink_bi_encoder": 
            self.retriever = BlinkBiEncoder(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.module_config["snapshot"]
            )
            # Build the index based on the pre-computed embeddings
            self.retriever.make_index(use_precomputed_entity_vectors=True)
        else:
            raise Exception(f"Invalid method: {self.module_config['method']}")

    def search(
        self,
        document: Document,
        num_candidate_entities: int = 10
    ) -> tuple[Document, CandidateEntitiesForDocument]:
        # Apply the retriever to the document
        return self.retriever.search(
            document=document,
            retrieval_size=num_candidate_entities
        )

