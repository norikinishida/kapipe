from __future__ import annotations

import logging

from tqdm import tqdm

from ..datatypes import (
    Document,
    CandidateEntitiesForDocument
)
from .base import BaseEDReranker


logger = logging.getLogger(__name__)


class IdenticalEntityReranker(BaseEDReranker):

    def __init__(self) -> None:
        logger.info(
            "########## IdenticalEntityReranker Initialization Starts ##########"
        )
        logger.info(
            "########## IdenticalEntityReranker Initialization Ends ##########"
        )

    def rerank(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument
    ) -> Document:
        return document
