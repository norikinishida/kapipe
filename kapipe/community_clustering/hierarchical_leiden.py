from __future__ import annotations

import logging
from typing import Any, cast

import networkx as nx
from graspologic.partition import hierarchical_leiden
from graspologic.utils import largest_connected_component
# import html

from ..datatypes import CommunityRecord


logger = logging.getLogger(__name__)


class HierarchicalLeiden:
    
    def __init__(self):
        pass

    def cluster_communities(
        self,
        graph: nx.MultiDiGraph,
        max_cluster_size: int = 10,
        use_lcc: bool = False 
    ) -> list[CommunityRecord]:
        logger.info("Applying Hierarchical Leiden algorithm ...")

        # Transform the graph to undirected one
        modified_graph = nx.DiGraph(graph).to_undirected()

        if use_lcc:
            modified_graph = self._stable_largest_connected_component(graph=modified_graph)

        # Apply Hierarchical Leiden algorithm
        community_mapping = hierarchical_leiden(
            graph=modified_graph,
            max_cluster_size=max_cluster_size
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
            if not community_id in communities:
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
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)

        # If the graph is undirected, we create the edges in a stable way, so we get the same results
        # for example:
        # A -> B
        # in graph theory is the same as
        # B -> A
        # in an undirected graph
        # however, this can lead to downstream issues because sometimes
        # consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
        # but they base some of their logic on the order of the nodes, so the order ends up being important
        # so we sort the nodes in the edge in a stable way, so that we always get the same order

        def _sort_edge(edge: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            u, v, data = edge
            return (min(u, v), max(u, v), data)

        def _edge_key(u: Any, v: Any) -> str:
            return f"{u} -> {v}"        

        edges = list(graph.edges(data=True)) 
        if not graph.is_directed():
            edges = [_sort_edge(e) for e in edges]

        edges = sorted(edges, key=lambda e: _edge_key(e[0], e[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

