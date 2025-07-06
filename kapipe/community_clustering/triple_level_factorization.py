from __future__ import annotations

import logging

import networkx as nx

from ..datatypes import CommunityRecord


logger = logging.getLogger(__name__)


class TripleLevelFactorization:
    """
    Cluster each triple (head, relation, tail) as an individual community.

    This method treats every edge in the graph as a unit and creates
    one community per triple, containing both head and tail nodes.
    """    

    def __init__(self):
        pass

    def cluster_communities(self, graph: nx.MultiDiGraph) -> list[CommunityRecord]:
        """
        Cluster the graph by treating each triple (head, relation, tail) as a community.
        """
        logger.info("Applying Triple-Level Factorization ...")

        # Initialize the community records
        communities: list[CommunityRecord] = []

        for head, tail, data in graph.edges(data=True):
            # Extract the relation label from the edge data
            relation = data["relation"]

            # Add a new community record for this triple
            communities.append({
                "community_id": f"Community({head},{relation},{tail})",
                "nodes": [head, tail],
                "level": 0,
                "parent_community_id": "ROOT",
                "child_community_ids": []
            })

        # Add a virtual root community record
        root_community = {
            "community_id": "ROOT",
            "nodes": None,
            "level": -1,
            "parent_community_id": None,
            "child_community_ids": [c["community_id"] for c in communities]
        }

        return [root_community] + communities

