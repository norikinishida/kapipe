from __future__ import annotations

import logging

import networkx as nx

from ..datatypes import CommunityRecord


logger = logging.getLogger(__name__)


class NeighborhoodAggregation:
    """
    A community detection algorithm that aggregates each node with its in- and out-neighbors.

    Each node forms a community with its in- and out-neighbors.
    This results in overlapping communities (if not deduplicated),
    with a flat hierarchy under a ROOT community.
    """ 

    def __init__(
        self,
        hop_size: int = 1
    ):
        # Validate the hop size
        if hop_size < 1:
            raise ValueError(f"hop_size must be >= 1: {hop_size}")

        self.hop_size = hop_size

    def cluster_communities(
        self,
        graph: nx.MultiDiGraph
    ) -> list[CommunityRecord]:
        """
        Apply the Neighborhood Aggregation to cluster communities in a directed graph.
        """

        logger.info("Applying Neighborhood Aggregation ...")

        # Initialize the community records
        communities: list[CommunityRecord] = []

        # Convert the directed graph to an undirected graph to preserve the current both-direction behavior
        undirected_graph = graph.to_undirected()

        for center_node in graph.nodes:
            #-----
            # [old implication] Get in- and out-neighbor nodes
            # out_neighbor_nodes = set(graph.neighbors(center_node))
            # in_neighbor_nodes = set(graph.predecessors(center_node))

            # Merge the neighbor nodes
            # neighbor_nodes = list(out_neighbor_nodes | in_neighbor_nodes)
            #-----

            #-----
            # Collect nodes reachable within the specified number of hops
            nodes_within_k_hops = nx.single_source_shortest_path_length(
                undirected_graph,
                center_node,
                cutoff=self.hop_size
            ).keys()

            # Remove the center node from the neighbor nodes
            neighbor_nodes = set(nodes_within_k_hops) - {center_node}

            # Sort the neighbor nodes to make the output deterministic
            neighbor_nodes = sorted(neighbor_nodes)
            #-----

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

