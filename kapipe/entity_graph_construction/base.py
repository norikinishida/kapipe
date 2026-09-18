from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx

from ..datatypes import Document, EntityPage


class BaseEntityGraphConstructor(ABC):
    """Base class for Entity Graph Construction components."""

    @abstractmethod
    def construct_entity_graph(
        self,
        documents: list[Document] | None,
        entity_dict: list[EntityPage] | None,
        additional_triples: list[dict[str, Any]] | None,
    ) -> nx.MultiDiGraph:
        """Construct an entity graph."""