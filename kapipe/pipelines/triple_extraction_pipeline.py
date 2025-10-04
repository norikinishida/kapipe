from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from ..datatypes import (
    Document,
)
from ..chunking import Chunker
from ..ner import NER
from ..ed_retrieval import EDRetrieval
from ..ed_reranking import EDReranking
from ..docre import DocRE


logger = logging.getLogger(__name__)


class TripleExtractionPipeline:

    def __init__(
        self,
        component_kwargs: dict[str, dict[str, Any]],
        share_backborn_llm: bool = False
    ):
        self.component_kwargs = component_kwargs
        self.share_backborn_llm = share_backborn_llm

        # Chunking
        if "chunking" in self.component_kwargs:
            self.chunker = Chunker(model_name=self.component_kwargs["chunking"]["model_name"])
        else:
            self.chunker = Chunker()

        # NER
        self.ner = NER(
            identifier=self.component_kwargs["ner"]["identifier"],
            gpu=self.component_kwargs["ner"].get("gpu", 0),
            entity_types=self.component_kwargs["ner"].get("entity_types", None)
        )

        if self.share_backborn_llm:
            llm_model = self.ner.extractor.model
        else:
            llm_model = None

        # ED-Retrieval
        self.ed_retrieval = EDRetrieval(
            identifier=self.component_kwargs["ed_retrieval"]["identifier"],
            gpu=self.component_kwargs["ed_retrieval"].get("gpu", 0)
        )

        # ED-Reranking
        self.ed_reranking = EDReranking(
            identifier=self.component_kwargs["ed_reranking"]["identifier"],
            gpu=self.component_kwargs["ed_reranking"].get("gpu", 0),
            llm_model=llm_model,
        )

        # DocRE
        self.docre = DocRE(
            identifier=self.component_kwargs["docre"]["identifier"],
            gpu=self.component_kwargs["docre"].get("gpu", 0),
            relation_labels=self.component_kwargs["docre"].get("relation_labels", None),
            llm_model=llm_model
        )

    def text_to_document(
        self,
        doc_key: str,
        text: str,
        title: str | None = None
    ) -> Document:
        return self.chunker.convert_text_to_document(
            doc_key=doc_key,
            text=text,
            title=title
        )
      
    def extract(self, document: Document, num_candidate_entities: int = 10) -> Document:
        # NER
        document = self.ner.extract(document=document)

        # ED-Retrieval
        document, candidate_entities_for_doc = self.ed_retrieval.search(
            document=document,
            num_candidate_entities=num_candidate_entities
        )

        # ED-Reranking
        document = self.ed_reranking.rerank(
            document=document,
            candidate_entities_for_doc=candidate_entities_for_doc
        )

        # DocRE
        document = self.docre.extract(document=document)

        return document

