from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import CandidateEntitiesForDocument, Document


class BaseEDReranker(ABC):
    """Base class for entity disambiguation reranking components."""

    @abstractmethod
    def rerank(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument
    ) -> Document:
        """Rerank candidate entities for each mention."""