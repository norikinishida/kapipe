from __future__ import annotations

import logging
from typing import Any, cast

import networkx as nx
from graspologic.partition import hierarchical_leiden
from graspologic.utils import largest_connected_component
# import html

from ..datatypes import CommunityRecord
from .base import BaseCommunityClusterer


logger = logging.getLogger(__name__)


class HierarchicalLeiden(BaseCommunityClusterer):
    """
    A hierarchical community detection algorithm based on the Leiden method.

    This results in a hierarchical structure of communities under a ROOT community.
    """
    
    def __init__(
        self,
        max_cluster_size: int = 10,
        use_lcc: bool = False ,
    ):
        self.max_cluster_size = max_cluster_size
        self.use_lcc = use_lcc

    def cluster_communities(
        self,
        graph: nx.MultiDiGraph,
    ) -> list[CommunityRecord]:
        """Apply the Hierarchical Leiden algorithm to cluster communities in a directed graph."""

        logger.info("Applying Hierarchical Leiden algorithm ...")

        # Transform the graph to undirected one
        modified_graph = nx.DiGraph(graph).to_undirected()

        # If requested, extract the largest connected component
        if self.use_lcc:
            modified_graph = self._stable_largest_connected_component(graph=modified_graph)

        # Apply Hierarchical Leiden algorithm
        community_mapping = hierarchical_leiden(
            graph=modified_graph,
            max_cluster_size=self.max_cluster_size
        )
        logger.info(f"Obtained {len(community_mapping)} communities")

        # Initialize the community records
        communities: dict[str, CommunityRecord] = {}

        for partition in community_mapping:
            # Get attributes of each node
            node_id = str(partition.node)
            community_id = str(partition.cluster)
            parent_id = (
                str(partition.parent_cluster)
                if partition.parent_cluster is not None else "ROOT"
            )
            level = int(partition.level)

            # Add a new community record
            if community_id not in communities:
                communities[community_id] = {
                    "community_id": community_id,
                    "nodes": [],
                    "level": level,
                    "parent_community_id": parent_id,
                    "child_community_ids": [],
                }

            # Add this node to the existing community record
            communities[community_id]["nodes"].append(node_id)

        # Add the ROOT community record
        communities["ROOT"] = {
            "community_id": "ROOT",
            "nodes": None,
            "level": -1,
            "parent_community_id": None,
            "child_community_ids": []
        }

        # Add parent-child relationships
        for community_id, community in communities.items():
            if community_id == "ROOT":
                continue
            parent_id = community["parent_community_id"]
            communities[parent_id]["child_community_ids"].append(community_id)

        # Sort the community records  based on the depth level
        communities = list(communities.values())
        communities = sorted(communities, key=lambda c: c["level"])

        return communities

    def _stable_largest_connected_component(self, graph: nx.Graph) -> nx.Graph:
        """Extract the largest connected component of the graph and stabilize it."""

        lcc = cast("nx.Graph", largest_connected_component(graph.copy()))
        # lcc = _normalize_node_names(graph=lcc)
        return self._stabilize_graph(graph=lcc)

    # def _normalize_node_names(self, graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    #     """Normalize node names."""
    #     node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
    #     return nx.relabel_nodes(graph, node_mapping)

    def _stabilize_graph(self, graph: nx.Graph) -> nx.Graph:
        """
        Ensure consistent ordering of nodes and edges in undirected graphs.

        Useful to avoid random node orderings which may affect downstream processing.
        """
        # Create a new graph with the same type (directed or undirected) as the input graph
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        # Add nodes to the new graph in a sorted order
        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])
        fixed_graph.add_nodes_from(sorted_nodes)

        def _sort_edge(edge: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            """Sort the nodes in an edge to ensure consistent ordering."""
            u, v, data = edge
            return (min(u, v), max(u, v), data)

        def _edge_key(u: Any, v: Any) -> str:
            """Create a stable key for an edge based on its nodes."""
            return f"{u} -> {v}"        

        # Add edges to the new graph in a sorted order
        edges = list(graph.edges(data=True)) 
        if not graph.is_directed():
            edges = [_sort_edge(e) for e in edges]
        edges = sorted(edges, key=lambda e: _edge_key(e[0], e[1]))
        fixed_graph.add_edges_from(edges)

        return fixed_graph

