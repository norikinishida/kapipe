from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx

from ..datatypes import Passage


class BasePassageGraphConstructor(ABC):
    """Base class for Passage Graph Construction components."""

    @abstractmethod
    def construct_passage_graph(
        self,
        passages: list[Passage],
        triples: list[dict[str, Any]],
        node_id_key: str,
    ) -> nx.DiGraph:
        """Construct a directed passage graph."""
