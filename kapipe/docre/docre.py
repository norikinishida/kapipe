from __future__ import annotations

import logging
import os
from os.path import expanduser
from typing import Any

from ..datatypes import (
    Config,
    Document,
    DemonstrationsForOneExample
)
from .. import utils
from ..demonstration_retrieval import DemonstrationRetriever
from .atlop import ATLOP
from .llm_docre import LLMDocRE


logger = logging.getLogger(__name__)


class DocRE:

    def __init__(
        self,
        identifier: str,
        gpu: int = 0,
        relation_labels: list[dict[str,str]] | None = None,
        llm_model: Any = None
    ):
        self.identifier = identifier
        self.gpu = gpu
        self.relation_labels = relation_labels

        root_config: Config = utils.get_hocon_config(
            os.path.join(expanduser("~"), ".kapipe", "download", "config")
        )
        self.component_config: Config = root_config["docre"][identifier]

        # # Download the configurations
        # utils.download_folder_if_needed(
        #     dest=self.component_config["snapshot"],
        #     url=self.component_config["url"]
        # )

        # Initialize the DocRE extractor
        if self.component_config["method"] == "atlop":
            self.extractor = ATLOP(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.component_config["snapshot"]
            )
        elif self.component_config["method"] == "llm_docre":
            if relation_labels is None:
                # Use pre-defined relation labels corresponding to the identifier
                self.extractor = LLMDocRE(
                    device=f"cuda:{self.gpu}",
                    path_snapshot=self.component_config["snapshot"],
                    model=llm_model,
                )
            else:
                # Use the user-defined relation labels
                self.extractor = LLMDocRE(
                    device=f"cuda:{self.gpu}",
                    vocab_relation = {
                        x["relation_label"]: i
                        for i, x in enumerate(relation_labels)
                    },
                    rel_meta_info = {
                        x["relation_label"]: {
                            "Pretty Name": x["relation_label"],
                            "Definition": x["definition"]
                        }
                        for x in relation_labels
                    },
                    path_snapshot=self.component_config["snapshot"],
                    model=llm_model,
                )

            # Initialize the demonstration retriever
            self.demonstration_retriever = DemonstrationRetriever(
                path_demonstration_pool=self.extractor.prompt_processor.path_demonstration_pool,
                method="count",
                task="docre"
            )
        else:
            raise Exception(f"Invalid method: {self.component_config['method']}")

    def extract(self, document: Document) -> Document:
        if self.component_config["method"] == "llm_docre":
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
