from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx

from ..datatypes import CommunityRecord, Passage


class BaseReportGenerator(ABC):
    """Base class for Report Generation components."""

    @abstractmethod
    def generate_community_reports(
        self,
        graph: nx.MultiDiGraph,
        communities: list[CommunityRecord],
        node_attr_keys: tuple[str, ...],
        edge_attr_keys: tuple[str, ...]
    ) -> list[Passage]:
        """Generate reports for communities."""