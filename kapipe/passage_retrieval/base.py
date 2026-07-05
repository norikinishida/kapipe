from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import Passage


class BasePassageRetriever(ABC):
    """Base class for Passage Retrieval components."""

    @abstractmethod
    def make_index(
        self,
        passages: list[Passage],
        index_dir: str,
    ) -> None:
        """Build an index from passages."""

    @abstractmethod
    def load_index(
        self,
        index_dir: str,
    ) -> None:
        """Load an existing index."""

    @abstractmethod
    def search(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[list[Passage]]:
        """Retrieve passages for each query."""