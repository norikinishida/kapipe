from __future__ import annotations

import logging

import networkx as nx

from ..datatypes import CommunityRecord
from .base import BaseCommunityClusterer


logger = logging.getLogger(__name__)


class TripleLevelFactorization(BaseCommunityClusterer):
    """
    A community detection algorithm that treats each triple (head, relation, tail) in a directed graph as a community.


    This method treats every edge in the graph as a unit and creates
    one community per triple, containing both head and tail nodes.
    """    

    def __init__(self):
        pass

    def cluster_communities(
        self,
        graph: nx.MultiDiGraph
    ) -> list[CommunityRecord]:
        """
        Apply the Triple-Level Factorization to cluster communities in a directed graph.
        """

        logger.info("Applying Triple-Level Factorization ...")

        # Initialize the community records
        communities: list[CommunityRecord] = []

        for head, tail, data in graph.edges(data=True):
            # Extract the relation label from the edge data
            relation = data["relation"]

            # Add a new community record for this triple
            communities.append({
                "community_key": f"Community({head},{relation},{tail})",
                "nodes": [head, tail],
                "level": 0,
                "parent_community_key": "ROOT",
                "child_community_keys": []
            })

        # Add a virtual root community record
        root_community = {
            "community_key": "ROOT",
            "nodes": None,
            "level": -1,
            "parent_community_key": None,
            "child_community_keys": [c["community_key"] for c in communities]
        }

        return [root_community] + communities
