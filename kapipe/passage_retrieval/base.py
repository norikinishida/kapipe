from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import Passage


class BasePassageRetriever(ABC):
    """Base class for passage retrieval components."""

    @abstractmethod
    def search(
        self,
        queries: list[str],
        top_k: int = 1
    ) -> list[list[Passage]]:
        """Retrieve passages for each query."""