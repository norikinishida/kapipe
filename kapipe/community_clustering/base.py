from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx

from ..datatypes import CommunityRecord


class BaseCommunityClusterer(ABC):
    """Base class for Community Clustering components."""

    @abstractmethod
    def cluster_communities(
        self,
        graph: nx.MultiDiGraph
    ) -> list[CommunityRecord]:
        """Cluster communities in a graph."""