from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import Document


class BaseNER(ABC):
    """Base class for NER components."""

    @abstractmethod
    def extract(self, document: Document) -> Document:
        """Extract named entity mentions from a document."""