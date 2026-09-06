from __future__ import annotations

import logging
import re
from typing import Any

import networkx as nx

from .base import BaseGraphRetriever


# XML 1.0 characters that cannot be represented in GraphML
_INVALID_XML_CHARS_RE = re.compile(
    "[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd"
    "\U00010000-\U0010ffff]"
)


logger = logging.getLogger(__name__)


class GraphRetriever(BaseGraphRetriever):
    """Retrieve a neighborhood subgraph around anchor nodes."""

    def __init__(
        self,
        # Optional
        use_timestamp: bool = False,
    ) -> None:

        self.use_timestamp = use_timestamp

        # Initialize the graph indexes before make_index() is called
        self.graph: nx.DiGraph | None = None
        self.undirected_graph: nx.Graph | None = None

    def make_index(
        self,
        graph: nx.DiGraph,
    ) -> None:
        """Build an undirected search index from a directed graph."""

        # Preserve the original directed graph for returning directed edges
        self.graph = graph

        # Build an undirected view for bidirectional neighborhood expansion
        self.undirected_graph = graph.to_undirected()

    def search(
        self,
        anchor_node_ids: list[str],
        hop_size: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Retrieve nodes and directed edges within the requested hop size."""

        # Reject an invalid neighborhood depth
        if hop_size < 0:
            raise ValueError("hop_size must be greater than or equal to 0.")

        # Require both graph representations to be initialized
        if self.graph is None or self.undirected_graph is None:
            raise ValueError(
                "Graph is not initialized. Call make_index(graph) first."
            )

        # Normalize anchor identifiers in the same way as GraphML node IDs
        normalized_anchor_node_ids = [
            sanitize_graphml_string(node_id)
            for node_id in anchor_node_ids
        ]
        anchor_node_id_set = set(normalized_anchor_node_ids)

        # Collect nodes reachable from at least one anchor node
        reachable_node_ids = self.get_reachable_node_ids(
            anchor_node_ids=normalized_anchor_node_ids,
            hop_size=hop_size,
        )

        # Return an empty subgraph when no valid anchor reaches the graph
        if len(reachable_node_ids) == 0:
            return [], []

        # Preserve the indexed graph's node order for reproducible results
        ordered_reachable_node_ids = [
            node_id
            for node_id in self.graph.nodes
            if node_id in reachable_node_ids
        ]

        # Collect node attributes for every reachable node
        reachable_nodes: dict[str, dict[str, Any]] = {}
        for node_id in ordered_reachable_node_ids:
            node_attributes = self.graph.nodes[node_id]
            reachable_nodes[node_id] = {
                "node_id": node_id,
                **node_attributes,
                "is_anchor": node_id in anchor_node_id_set,
            }

        # Collect directed edge attributes internal to the reachable nodes
        internal_edges: dict[tuple[str, str], dict[str, Any]] = {}
        for head_id in ordered_reachable_node_ids:
            for tail_id in ordered_reachable_node_ids:
                # Exclude self-loops as in the published implementation
                if head_id == tail_id:
                    continue

                # Skip node pairs without an original directed edge
                if (head_id, tail_id) not in self.graph.edges:
                    continue

                # Read the node and edge attributes required by the result
                head_attributes = self.graph.nodes[head_id]
                tail_attributes = self.graph.nodes[tail_id]
                edge_attributes = self.graph.edges[head_id, tail_id]

                # Preserve the published nested endpoint representation
                edge: dict[str, Any] = {
                    "head": {
                        "node_id": head_id,
                        **head_attributes,
                        "is_anchor": head_id in anchor_node_id_set,
                    },
                    "tail": {
                        "node_id": tail_id,
                        **tail_attributes,
                        "is_anchor": tail_id in anchor_node_id_set,
                    },
                    "relation": edge_attributes["relation"],
                }

                # Preserve the optional relation explanation when available
                if "explanation" in edge_attributes:
                    edge["explanation"] = edge_attributes["explanation"]

                # Keep one edge for each directed node pair
                internal_edges[(head_id, tail_id)] = edge

        # Convert the collected mappings into result lists
        reachable_node_list = list(reachable_nodes.values())
        internal_edge_list = list(internal_edges.values())

        # Reorder the results only when timestamps are enabled
        if self.use_timestamp:
            reachable_node_list, internal_edge_list = (
                self.reorder_nodes_and_edges_using_timestamp(
                    nodes=reachable_node_list,
                    edges=internal_edge_list,
                )
            )

        # Replace redundant edge endpoints with indices into the node list
        reachable_node_list, internal_edge_list = self.post_process(
            nodes=reachable_node_list,
            edges=internal_edge_list,
        )

        return reachable_node_list, internal_edge_list

    def get_reachable_node_ids(
        self,
        anchor_node_ids: list[str],
        hop_size: int,
    ) -> set[str]:
        """Collect graph nodes reachable from the anchor nodes."""

        # Require the undirected search index to be initialized
        if self.undirected_graph is None:
            raise ValueError(
                "Graph is not initialized. Call make_index(graph) first."
            )

        # Initialize the union of neighborhoods from all valid anchors
        reachable_node_ids: set[str] = set()

        # Expand each anchor independently on the undirected graph
        for anchor_node_id in anchor_node_ids:
            # Ignore anchors absent from the indexed graph
            if anchor_node_id not in self.undirected_graph:
                logger.warning(
                    "Anchor node is not in the graph: %s",
                    anchor_node_id,
                )
                continue

            # Retrieve shortest paths up to the requested hop size
            paths = nx.single_source_shortest_path(
                self.undirected_graph,
                anchor_node_id,
                cutoff=hop_size,
            )

            # Merge the reached node identifiers across all anchors
            reachable_node_ids.update(paths.keys())

        return reachable_node_ids

    def reorder_nodes_and_edges_using_timestamp(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Reorder timestamped nodes and their internal edges."""

        # Create a mapping from each node identifier to its node record
        node_id_to_node = {
            node["node_id"]: node
            for node in nodes
        }

        # Reconstruct the retrieved directed graph from the edge records
        graph = nx.DiGraph()
        graph.add_nodes_from(node_id_to_node.keys())
        for edge in edges:
            head_id = edge["head"]["node_id"]
            tail_id = edge["tail"]["node_id"]
            graph.add_edge(head_id, tail_id)

        # Store fixed timestamp and successor information for every node
        node_id_to_base_information: dict[str, dict[str, Any]] = {}
        for node_id in graph.nodes():
            timestamp = node_id_to_node[node_id]["timestamp"]
            tail_ids = list(graph.successors(node_id))
            node_id_to_base_information[node_id] = {
                "timestamp": timestamp,
                "tail_count": len(tail_ids),
                "tail_ids": tail_ids,
            }

        # Build the initial order by timestamp, successor count, and node ID
        sorted_node_ids = sorted(
            graph.nodes(),
            key=lambda node_id: (
                node_id_to_base_information[node_id]["timestamp"],
                node_id_to_base_information[node_id]["tail_count"],
                node_id,
            ),
        )

        # Refine the order using successor positions as in the published code
        for _ in range(3):
            node_id_to_index = {
                node_id: node_index
                for node_index, node_id in enumerate(sorted_node_ids)
            }

            # Compute the sum of the positions of each node's successors in the current order
            node_id_to_tail_index_sum: dict[str, float | int] = {}
            for node_id in sorted_node_ids:
                tail_ids = node_id_to_base_information[node_id]["tail_ids"]
                tail_indices = [
                    node_id_to_index[tail_id]
                    for tail_id in tail_ids
                    if tail_id in node_id_to_index
                ]
                node_id_to_tail_index_sum[node_id] = (
                    sum(tail_indices)
                    if len(tail_indices) > 0
                    else float("inf")
                )

            # Preserve stable ordering while refining successor proximity
            sorted_node_ids = sorted(
                sorted_node_ids,
                key=lambda node_id: (
                    node_id_to_base_information[node_id]["timestamp"],
                    node_id_to_base_information[node_id]["tail_count"],
                    node_id_to_tail_index_sum[node_id],
                    node_id,
                ),
            )

        # Apply the final order to the node records
        ordered_nodes = [
            node_id_to_node[node_id]
            for node_id in sorted_node_ids
        ]

        # Apply the corresponding endpoint order to the edge records
        node_id_to_position = {
            node_id: node_index
            for node_index, node_id in enumerate(sorted_node_ids)
        }
        ordered_edges = sorted(
            edges,
            key=lambda edge: (
                node_id_to_position[edge["head"]["node_id"]],
                node_id_to_position[edge["tail"]["node_id"]],
            ),
        )

        return ordered_nodes, ordered_edges

    def post_process(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Replace edge endpoint records with node-list indices."""

        # Create a mapping from each node identifier to its final list index
        node_id_to_index = {
            node["node_id"]: node_index
            for node_index, node in enumerate(nodes)
        }

        # Replace redundant endpoint records with references to the node list
        for edge_index in range(len(edges)):
            edge = edges[edge_index]
            head_id = edge["head"]["node_id"]
            tail_id = edge["tail"]["node_id"]
            edge["head"] = node_id_to_index[head_id]
            edge["tail"] = node_id_to_index[tail_id]
            edges[edge_index] = edge

        return nodes, edges


def sanitize_graphml_string(value: str) -> str:
    """Remove characters that XML 1.0 cannot represent."""

    # Remove invalid XML characters without changing valid content
    return _INVALID_XML_CHARS_RE.sub("", value)
