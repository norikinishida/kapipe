from __future__ import annotations

import logging

import networkx as nx

from ..datatypes import (
    CommunityRecord,
    Passage
)


logger = logging.getLogger(__name__)


class TemplateBasedReportGenerator:
    
    def __init__(self):
        pass

    def generate_community_reports(
        self,
        # Input
        graph: nx.MultiDiGraph,
        communities: list[CommunityRecord],
        node_attr_keys: tuple[str, ...],
        edge_attr_keys: tuple[str, ...],
        # Misc.
        relation_map: dict[str, str] | None = None
    ) -> list[Passage]:
        """Generate reports using a deterministic template instead of LLM."""

        assert len(node_attr_keys) > 0
        assert len(edge_attr_keys) > 0

        if relation_map is None:
            relation_map = {}
    
        n_total = len(communities) - 1 # Exclude ROOT
        count = 0

        reports: list[Passage] = []

        for community in communities:
            # Skip ROOT
            if community["community_id"] == "ROOT":
                continue

            count += 1

            # Get nodes that belong to this community directly
            direct_nodes = community["nodes"]

            logger.info(f"[{count}/{n_total}] Generating a report for community ({community['community_id']}) with {len(direct_nodes)} direct nodes ...")

            # Limit number of direct nodes
            if len(direct_nodes) >= 100:
                logger.info(f"[{count}/{n_total}] Reducing nodes to top 100 primary nodes among {len(direct_nodes)} nodes")
                direct_nodes = [
                    n for n, d in sorted(
                        graph.subgraph(direct_nodes).degree(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:100]
                ]

            # Get edges for this community
            edges = [
                (h,t,p) for h,t,p in graph.edges(direct_nodes, data=True)
                if (h in direct_nodes) and (t in direct_nodes)
            ]

            # Get top 3 primary nodes for this community
            key_nodes = [
                n for n, d in sorted(
                    graph.subgraph(direct_nodes).degree(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ]
            # key_node_names = [graph.nodes[n]["name"] for n in key_nodes]
            key_node_names = [
                graph.nodes[n][node_attr_keys[0]].replace("|", " ").replace("\n", " ").strip()
                for n in key_nodes
            ]

            # Fill the title
            # content_title = f"The primary entities in this community are: {', '.join(key_node_names)}"
            content_title = f"The primary nodes in this community are: {', '.join(key_node_names)}"

            # Fill the content text
            content_text = ""
            if len(direct_nodes) > 0:
                # content_text += "This community contains the following entities:\n"
                content_text += "This community contains the following nodes:\n"
                for node in direct_nodes:
                    props = graph.nodes[node]

                    # name = props["name"].replace("|", " ")
                    # etype = props["entity_type"].replace("|", " ")
                    # desc = props["description"].replace("|", " ").replace("\n", " ").rstrip()
                    # if desc == "":
                    #     desc = "N/A"
                    # content_text += f"- {name} | {etype} | {desc}\n"

                    # Create one line based on node_attr_keys
                    values = []
                    for key in node_attr_keys:
                        v = props[key].replace("|", " ").replace("\n", " ").strip()
                        if key == "description" and v == "":
                            v = "N/A"
                        values.append(v)
                    line = " | ".join(values)
                    content_text += f"- {line}\n"

            if len(edges) > 0:
                # content_text += "The relationships between the entities are as follows:\n"
                content_text += "The relationships between the nodes are as follows:\n"
                for head, tail, props in edges:
                    # head_name = graph.nodes[head]["name"].replace("|", " ")
                    # tail_name = graph.nodes[tail]["name"].replace("|", " ")
                    # relation = props["relation"]
                    # relation = relation_map.get(relation, relation).replace("|", " ")
                    # content_text += f"- {head_name} | {relation} | {tail_name}\n"

                    # Create one line based on edge_attr_keys
                    head_name = graph.nodes[head][node_attr_keys[0]].replace("|", " ").replace("\n", " ").strip()
                    tail_name = graph.nodes[tail][node_attr_keys[0]].replace("|", " ").replace("\n", " ").strip()
                    relation = props[edge_attr_keys[0]]
                    relation = relation_map.get(relation, relation).replace("|", " ").replace("\n", " ").strip()
                    values = [head_name, relation, tail_name]
                    if len(edge_attr_keys) > 1:
                        for key in edge_attr_keys[1:]:
                            v = props[key].replace("|", " ").replace("\n", " ").strip()
                            values.append(v)
                    line = " | ".join(values)
                    content_text += f"- {line}\n"

            # Finalize the report
            report = {
                "title": content_title,
                "text": content_text.strip()
            } | community

            # Record the resulting report
            reports.append(report)

        return reports
