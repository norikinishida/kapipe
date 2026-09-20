from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx


class BaseGraphRetriever(ABC):
    """Base class for Graph Retriever components."""

    @abstractmethod
    def make_index(
        self,
        graph: nx.DiGraph,
    ) -> None:
        """Build an index from a directed graph."""

    @abstractmethod
    def search(
        self,
        anchor_node_ids: list[str],
        hop_size: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Retrieve nodes and index-linked edges around anchor nodes."""
