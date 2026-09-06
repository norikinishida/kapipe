from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import Passage


class BasePropositionExtractor(ABC):
    """Base class for Proposition Extraction components."""

    @abstractmethod
    def extract(
        self,
        passage: Passage,
    ) -> list[Passage]:
        """Extract propositions from a passage."""
