from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx


class BaseEntityGraphConstructor(ABC):
    """Base class for Entity Graph Construction components."""

    @abstractmethod
    def construct_entity_graph(
        self,
        documents_path_list: list[str] | None,
        entity_dict_path: str | None,
        additional_triples_path: str | None,
    ) -> nx.MultiDiGraph:
        """Construct an entity graph."""