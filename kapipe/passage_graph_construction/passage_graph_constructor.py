from __future__ import annotations

import re
from typing import Any

import networkx as nx

from ..datatypes import Passage
from .base import BasePassageGraphConstructor


# GraphML-compatible scalar attribute types
_GRAPHML_ALLOWED_TYPES = (str, int, float, bool)

# XML 1.0 characters that cannot be represented in GraphML
_INVALID_XML_CHARS_RE = re.compile(
    "[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd"
    "\U00010000-\U0010ffff]"
)


class PassageGraphConstructor(BasePassageGraphConstructor):
    """Construct a directed graph whose nodes represent passages."""

    def construct_passage_graph(
        self,
        passages: list[Passage],
        triples: list[dict[str, Any]],
        node_id_key: str,
    ) -> nx.DiGraph:
        """Construct a directed passage graph from passages and relation triples."""

        # Initialize a directed graph that keeps at most one edge per node pair
        graph = nx.DiGraph()

        # Add all passage nodes before processing relation triples
        for passage in passages:
            # Read and sanitize the required passage identifier
            passage_id = sanitize_graphml_string(passage[node_id_key])

            # Keep only attributes that GraphML can represent
            graph.add_node(
                passage_id,
                **sanitize_graphml_attributes(passage),
            )

        # Add relation edges in input order
        for triple in triples:
            # Read the passages, relation, and explanation stored in the triple
            head_passage = triple["head"]
            tail_passage = triple["tail"]
            relation = triple["relation"]
            explanation = triple.get("explanation")

            # Read and sanitize the required passage identifiers
            head_id = sanitize_graphml_string(head_passage[node_id_key])
            tail_id = sanitize_graphml_string(tail_passage[node_id_key])

            # Add a head passage that was not present in the input passage list
            if head_id not in graph:
                graph.add_node(
                    head_id,
                    **sanitize_graphml_attributes(head_passage),
                )

            # Add a tail passage that was not present in the input passage list
            if tail_id not in graph:
                graph.add_node(
                    tail_id,
                    **sanitize_graphml_attributes(tail_passage),
                )

            # Keep the first triple when the same directed node pair occurs again
            if graph.has_edge(head_id, tail_id):
                continue

            # Store the required relation as an edge attribute
            edge_attributes: dict[str, Any] = {
                "relation": relation,
            }

            # Store the optional explanation when it is available
            if explanation is not None:
                edge_attributes["explanation"] = explanation

            # Add the relation after making its attributes GraphML-compatible
            graph.add_edge(
                head_id,
                tail_id,
                **sanitize_graphml_attributes(edge_attributes),
            )

        return graph


def sanitize_graphml_string(value: str) -> str:
    """Remove characters that XML 1.0 cannot represent."""

    # Remove invalid XML characters without changing valid content
    return _INVALID_XML_CHARS_RE.sub("", value)


def sanitize_graphml_attributes(
    attributes: dict[str, Any],
) -> dict[str, Any]:
    """Keep and sanitize attributes that GraphML can represent."""

    # Initialize the GraphML-compatible attribute mapping
    sanitized_attributes: dict[str, Any] = {}

    # Inspect every source attribute independently
    for key, value in attributes.items():
        # Skip nested values and other unsupported GraphML attribute types
        if not isinstance(value, _GRAPHML_ALLOWED_TYPES):
            continue

        # Remove invalid XML characters from string values
        if isinstance(value, str):
            sanitized_attributes[key] = sanitize_graphml_string(value)
        else:
            # Preserve supported non-string scalar values
            sanitized_attributes[key] = value

    return sanitized_attributes
