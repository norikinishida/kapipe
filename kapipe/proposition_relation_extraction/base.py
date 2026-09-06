from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..datatypes import Passage


class BasePropositionRelationExtractor(ABC):
    """Base class for Proposition Relation Extraction components."""

    @abstractmethod
    def make_index(
        self,
        propositions: list[Passage],
        index_dir: str,
        **kwargs: Any,
    ) -> None:
        """Build a retrieval index from propositions."""

    @abstractmethod
    def load_index(
        self,
        index_dir: str,
    ) -> None:
        """Load an existing proposition retrieval index."""

    @abstractmethod
    def retrieve_tail_propositions(
        self,
        head_proposition: Passage,
        top_k: int,
        prefilter_k: int,
    ) -> list[Passage]:
        """Retrieve candidate tail propositions for a head proposition."""

    @abstractmethod
    def batch_retrieve_tail_propositions(
        self,
        head_propositions: list[Passage],
        top_k: int,
        prefilter_k: int,
        batch_size: int,
    ) -> list[list[Passage]]:
        """Retrieve candidate tail propositions for head propositions."""

    @abstractmethod
    def extract(
        self,
        head_proposition: Passage,
        tail_propositions: list[Passage],
    ) -> list[dict[str, Any]]:
        """Extract relations from a head proposition to tail propositions."""
