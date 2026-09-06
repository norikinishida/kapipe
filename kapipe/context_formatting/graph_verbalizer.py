from __future__ import annotations

from typing import Any

from .base import BaseContextFormatter


class GraphVerbalizer(BaseContextFormatter):
    """Verbalize graph nodes and edges as LLM-readable context."""

    def __init__(
        self,
        # Optional
        use_timestamp: bool = False,
    ) -> None:

        self.use_timestamp = use_timestamp

    def convert(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> str:
        """Verbalize graph nodes and index-linked edges."""

        # Define relation labels with specialized verbalizations
        # c.f., ProStruct-RAG specialized relation labels
        specialized_relation_labels = [
            "updates",
            "contradicts",
            "supports",
        ]

        # Preserve the specialized relation order and append unknown relations
        relation_label_keys = list(specialized_relation_labels)
        relation_label_key_to_relation_label = {
            relation_label: relation_label
            for relation_label in specialized_relation_labels
        }

        # Build outgoing and incoming relation maps for every node
        node_to_rel_to_dir_to_nodes: dict[
            int,
            dict[str, dict[str, list[int]]],
        ] = {}
        for node_i in range(len(nodes)):
            node_to_rel_to_dir_to_nodes[node_i] = {}

        for edge in edges:
            # Read the relation label without restricting its vocabulary
            relation_label = edge["relation"].strip()
            normalized_relation_label = relation_label.lower()

            # Normalize only the labels with specialized verbalizations
            if normalized_relation_label in specialized_relation_labels:
                relation_label_key = normalized_relation_label
            else:
                relation_label_key = relation_label

            # Preserve unknown relations in their first-seen order
            if relation_label_key not in relation_label_key_to_relation_label:
                relation_label_keys.append(relation_label_key)
                relation_label_key_to_relation_label[relation_label_key] = relation_label

            # Read the index-linked edge endpoints
            head_i = edge["head"]
            tail_i = edge["tail"]

            # Initialize the relation records only where they are needed
            for node_i in [head_i, tail_i]:
                if (
                    relation_label_key
                    not in node_to_rel_to_dir_to_nodes[node_i]
                ):
                    node_to_rel_to_dir_to_nodes[node_i][
                        relation_label_key
                    ] = {
                        "outgoing": [],
                        "incoming": [],
                    }

            # Record the directed relation from the head to the tail
            node_to_rel_to_dir_to_nodes[head_i][relation_label_key]["outgoing"].append(
                tail_i
            )
            node_to_rel_to_dir_to_nodes[tail_i][relation_label_key]["incoming"].append(
                head_i
            )

        # Verbalize the graph
        lines: list[str] = []

        lines.append("Structured Statements:")
        lines.append("")

        for node_i, node in enumerate(nodes):
            # Verbalize the proposition
            node_text = node["text"].strip()
            lines.append(f"[P{node_i+1}]")
            if self.use_timestamp:
                node_timestamp = node["timestamp"].strip()
                lines.append(f"Date: {node_timestamp}")
            lines.append(f"Statement: {node_text}")

            # Verbalize the relations connected to the proposition
            relation_lines: list[str] = []

            for relation_label_key in relation_label_keys:
                relation_label_information = (
                    node_to_rel_to_dir_to_nodes[node_i].get(
                        relation_label_key
                    )
                )
                if relation_label_information is None:
                    continue

                outgoing_node_indices = relation_label_information["outgoing"]
                incoming_node_indices = relation_label_information["incoming"]

                # Handle specialized relation labels differently
                if relation_label_key in specialized_relation_labels:
                    if relation_label_key == "updates":
                        passive_verb = "updated"
                    elif relation_label_key == "contradicts":
                        passive_verb = "contradicted"
                    else:
                        assert relation_label_key == "supports"
                        passive_verb = "supported"

                    if outgoing_node_indices:
                        if self.use_timestamp:
                            targets = ", ".join(
                                f"P{target_i+1} "
                                f"({nodes[target_i]['timestamp']})"
                                for target_i in outgoing_node_indices
                            )
                            relation_lines.append(
                                f"- This statement {relation_label_key} the following "
                                f"earlier propositions: {targets}."
                            )
                        else:
                            targets = ", ".join(
                                f"P{target_i+1}"
                                for target_i in outgoing_node_indices
                            )
                            relation_lines.append(
                                f"- This statement {relation_label_key} the following "
                                f"propositions: {targets}."
                            )

                    if incoming_node_indices:
                        if self.use_timestamp:
                            sources = ", ".join(
                                f"P{source_i+1} "
                                f"({nodes[source_i]['timestamp']})"
                                for source_i in incoming_node_indices
                            )
                            relation_lines.append(
                                f"- This statement is {passive_verb} by the "
                                f"following later propositions: {sources}."
                            )
                        else:
                            sources = ", ".join(
                                f"P{source_i+1}"
                                for source_i in incoming_node_indices
                            )
                            relation_lines.append(
                                f"- This statement is {passive_verb} by the "
                                f"following propositions: {sources}."
                            )

                    continue

                # Verbalize an unknown relation without inferring its semantics
                relation_label = relation_label_key_to_relation_label[relation_label_key]

                if outgoing_node_indices:
                    if self.use_timestamp:
                        targets = ", ".join(
                            f"P{target_i+1} "
                            f"({nodes[target_i]['timestamp']})"
                            for target_i in outgoing_node_indices
                        )
                    else:
                        targets = ", ".join(
                            f"P{target_i+1}"
                            for target_i in outgoing_node_indices
                        )

                    relation_lines.append(
                        f'- This statement has the "{relation_label}" relation '
                        f"to the following earlier propositions: {targets}."
                    )

                if incoming_node_indices:
                    if self.use_timestamp:
                        sources = ", ".join(
                            f"P{source_i+1} "
                            f"({nodes[source_i]['timestamp']})"
                            for source_i in incoming_node_indices
                        )
                    else:
                        sources = ", ".join(
                            f"P{source_i+1}"
                            for source_i in incoming_node_indices
                        )

                    relation_lines.append(
                        f'- The following later propositions have the '
                        f'"{relation_label}" relation to this statement: '
                        f"{sources}."
                    )

            if relation_lines:
                lines.append("Relations:")
                lines.extend(relation_lines)

            lines.append("")

        # Convert the graph into text
        text = "\n".join(lines).strip()

        return text
