from __future__ import annotations

import json
import logging

import networkx as nx

from ..datatypes import CommunityRecord


logger = logging.getLogger(__name__)


class TemplateBasedReportGenerator:
    
    def __init__(self):
        pass

    def generate_community_reports(
        self,
        # Input
        graph: nx.MultiDiGraph,
        communities: list[CommunityRecord],
        # Output processing
        path_output: str,
        # Relation label mapping
        relation_map: dict[str, str] | None = None
    ) -> None:
        """Generate reports using a deterministic template instead of LLM."""

        if relation_map is None:
            relation_map = {}
    
        n_total = len(communities) - 1 # Exclude ROOT
        count = 0

        with open(path_output, "w") as fout:
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
                key_node_names = [graph.nodes[n]["name"] for n in key_nodes]

                # Fill the title
                content_title = f"The primary entities in this community are: {', '.join(key_node_names)}"

                # Fill the content text
                content_text = ""
                if len(direct_nodes) > 0:
                    content_text += "This community contains the following entities:\n"
                    for node in direct_nodes:
                        props = graph.nodes[node]
                        name = props["name"].replace("|", " ")
                        etype = props["entity_type"].replace("|", " ")
                        desc = props["description"].replace("|", " ").replace("\n", " ").rstrip()
                        if desc == "":
                            desc = "N/A"
                        content_text += f"- {name} | {etype} | {desc}\n"
                if len(edges) > 0:
                    content_text += "The relationships between the entities are as follows:\n"
                    for head, tail, props in edges:
                        head_name = graph.nodes[head]["name"].replace("|", " ")
                        tail_name = graph.nodes[tail]["name"].replace("|", " ")
                        relation = props["relation"]
                        relation = relation_map.get(relation, relation).replace("|", " ")
                        content_text += f"- {head_name} | {relation} | {tail_name}\n"

                # Finalize the report
                report = {
                    "title": content_title,
                    "text": content_text.strip()
                } | community

                # Save the report
                json_str = json.dumps(report)           
                fout.write(json_str + "\n")


