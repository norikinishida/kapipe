from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import CandidateEntitiesForDocument, Document


class BaseEDRetriever(ABC):
    """Base class for entity disambiguation retrieval components."""

    @abstractmethod
    def make_index(self) -> None:
        """Build the index."""
 
    @abstractmethod
    def search(
        self,
        document: Document,
        retrieval_size: int = 1
    ) -> tuple[Document, CandidateEntitiesForDocument]:
        """Retrieve candidate entities for each mention."""