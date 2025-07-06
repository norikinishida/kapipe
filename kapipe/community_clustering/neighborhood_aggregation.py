from __future__ import annotations

import logging

import networkx as nx

from ..datatypes import CommunityRecord


logger = logging.getLogger(__name__)


class NeighborhoodAggregation:
    """
    Cluster nodes in a graph by simple neighborhood aggregation.

    Each node forms a community with its in- and out-neighbors.
    This results in overlapping communities (if not deduplicated),
    with a flat hierarchy under a ROOT community.
    """ 

    def __init__(self):
        pass

    def cluster_communities(self, graph: nx.MultiDiGraph) -> list[CommunityRecord]:
        """
        Cluster the graph using neighborhood aggregation strategy.
        """
        logger.info("Applying Neighborhood Aggregation ...")

        # Initialize the community records
        communities: list[CommunityRecord] = []

        for center_node in graph.nodes:
            # Get in- and out-neighbor nodes
            out_neighbor_nodes = set(graph.neighbors(center_node))
            in_neighbor_nodes = set(graph.predecessors(center_node))

            # Merge the neighbor nodes
            neighbor_nodes = list(out_neighbor_nodes | in_neighbor_nodes)

            # Add a new community record
            communities.append({
                "community_id": f"Community({center_node})",
                "nodes": [center_node] + neighbor_nodes,
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

