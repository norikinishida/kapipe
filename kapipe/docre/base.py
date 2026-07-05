from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import Document


class BaseDocRE(ABC):
    """Base class for Document-level Relation Extraction (DocRE) components."""

    @abstractmethod
    def extract(self, document: Document) -> Document:
        """Extract relations from a document."""