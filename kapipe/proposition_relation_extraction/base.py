from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..datatypes import Passage


class BasePropositionRelationExtractor(ABC):
    """Base class for Proposition Relation Extraction components."""

    @abstractmethod
    def extract(
        self,
        head_proposition: Passage,
        tail_propositions: list[Passage],
    ) -> list[dict[str, Any]]:
        """Extract relations from a head proposition to tail propositions."""
