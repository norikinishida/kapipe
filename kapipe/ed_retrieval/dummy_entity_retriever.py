from __future__ import annotations

import copy
import logging

from tqdm import tqdm

from .. import utils
from ..datatypes import (
    Document,
    Mention,
    Entity,
    CandidateEntitiesForDocument,
)


logger = logging.getLogger(__name__)


class DummyEntityRetriever:

    def __init__(self) -> None:
        logger.info("########## DummyEntityRetriever Initialization Starts ##########")
        logger.info("########## DummyEntityRetriever Initialization Ends ##########")

    def search(
        self,
        document: Document,
        retrieval_size: int = 1
    ) -> tuple[Document, CandidateEntitiesForDocument]:
        # Skip prediction if no mention appears
        if len(document["mentions"]) == 0:
           result_document = copy.deepcopy(document)
           result_document["entities"] = []
           candidate_entities_for_doc = {
              "doc_key": result_document["doc_key"],
              "candidate_entities": []
           }
           return result_document, candidate_entities_for_doc

        # Assign mention names as the Entity ID
        mentions: list[Mention] = []
        for mention in document["mentions"]:
            entity_id = mention["name"].lower()
            mentions.append({"entity_id": entity_id})

        # Transform to entity-level entity IDs
        entities: list[Entity] = utils.aggregate_mentions_to_entities(
            document=document,
            mentions=mentions
        )

        # Update the document
        result_document = copy.deepcopy(document)
        for i in range(len(result_document["mentions"])):
            result_document["mentions"][i].update(mentions[i])
        result_document["entities"] = entities
 
        # Transform to candidate entities for each mention
        candidate_entities_for_doc: CandidateEntitiesForDocument = {
            "doc_key": result_document["doc_key"],
            "candidate_entities": [
                [{"entity_id": m["entity_id"], "score": 1.0}] * retrieval_size
                for m in mentions
            ],
        }

        return result_document, candidate_entities_for_doc

    def batch_search(
        self,
        documents: list[Document],
        retrieval_size: int = 1
    ) -> tuple[list[Document], list[CandidateEntitiesForDocument]]:
        result_documents: list[Document] = []
        candidate_entities: list[CandidateEntitiesForDocument] = []
        for document in tqdm(documents, desc="retrieval steps"):
            result_document, candidate_entities_for_doc = self.search(
                document=document, retrieval_size=retrieval_size
            )
            result_documents.append(result_document)
            candidate_entities.append(candidate_entities_for_doc)
        return result_documents, candidate_entities
