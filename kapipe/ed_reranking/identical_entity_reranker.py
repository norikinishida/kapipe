from __future__ import annotations

import logging

from tqdm import tqdm

from ..datatypes import (
    Document,
    CandidateEntitiesForDocument
)


logger = logging.getLogger(__name__)


class IdenticalEntityReranker:

    def __init__(self):
        logger.info("########## IdenticalEntityReranker Initialization Starts ##########")
        logger.info("########## IdenticalEntityReranker Initialization Ends ##########")

    def rerank(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument
    ) -> Document:
        return document

    def batch_rerank(
        self,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument]
    ) -> list[Document]:
        result_documents = []
        for document, candidate_entities_for_doc in tqdm(
            zip(documents, candidate_entities),
            total=len(documents),
            desc="reranking steps"
        ):
            result_document = self.rerank(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc
            )
            result_documents.append(result_document)
        return result_documents

