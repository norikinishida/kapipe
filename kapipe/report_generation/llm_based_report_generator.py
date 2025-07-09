from __future__ import annotations

import json
import logging

import networkx as nx

from ..models import OpenAILLM
from .. import utils
from ..datatypes import (
    CommunityRecord,
    Passage
)


logger = logging.getLogger(__name__)


class LLMBasedReportGenerator:
    
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
        relation_map: dict[str, str] | None = None,
        parse_generated_text_fn = None
    ) -> None:
        """Generate reports for each community using an LLM."""

        if relation_map is None:
            relation_map = {}
    
        if parse_generated_text_fn is None:
            parse_generated_text_fn = parse_generated_text

        # Load prompt template for report generation
        prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path="report_generation_01_zeroshot"
        )

        # Initialize the LLM model
        model = OpenAILLM(openai_model_name="gpt-4o-mini", max_new_tokens=1024)

        # Instantiate the report generator
        report_generator = ReportGenerator(
            prompt_template=prompt_template,
            model=model,
            graph=graph,
            relation_map=relation_map,
            parse_generated_text_fn=parse_generated_text_fn,
            n_total=len(communities)-1 # Exclude ROOT
        )

        # Convert list of communities to dictionary for quick access
        communities_dict = {c["community_id"]: c for c in communities}

        # Generate community reports recursively in the bottom-up manner
        with open(path_output, "w") as fout:
            self._recursive(
                community=communities_dict["ROOT"],
                communities_dict=communities_dict,
                report_generator=report_generator,
                fout=fout
            )

    def _recursive(
        self,
        community: CommunityRecord,
        communities_dict: dict[str, CommunityRecord],
        report_generator: ReportGenerator,
        fout
    ) -> Passage | None:
        """Recursively generate reports in bottom-up fashion."""

        # Generate the sub-communities' reports recursively
        child_reports: list[Passage] = []
        for child_id in community["child_community_ids"]:
            if child_id in communities_dict:
                child_community = communities_dict[child_id]
                child_report = self._recursive(
                    community=child_community,
                    communities_dict=communities_dict,
                    report_generator=report_generator,
                    fout=fout
                )
                child_reports.append(child_report)

        # Skip ROOT
        if community["community_id"] == "ROOT":
            return None

        # Collect nodes that belong to this community directly
        nodes_of_children = utils.flatten_lists([c["nodes"] for c in child_reports])
        direct_nodes = [
            node for node in community["nodes"] if node not in nodes_of_children
        ]

        # Generate this community's report
        report = report_generator.generate_community_report(
            community=community,
            direct_nodes=direct_nodes,
            child_reports=child_reports
        )

        # Save the report
        json_str = json.dumps(report)
        fout.write(json_str + "\n")
        return report           

    
class ReportGenerator:
    
    def __init__(
        self,
        prompt_template: str,
        model: OpenAILLM,
        graph: nx.MultiDiGraph,
        relation_map: dict[str,str],
        parse_generated_text_fn,
        n_total: int
    ) -> None:
        self.prompt_template = prompt_template
        self.model = model
        self.graph = graph
        self.relation_map = relation_map
        self.parse_generated_text_fn = parse_generated_text_fn
        self.n_total = n_total
        self.count = 0

    def generate_community_report(
        self,
        community: CommunityRecord,
        direct_nodes: list[str],
        child_reports: list[Passage]
    ) -> Passage:
        """Generate a report for one community."""

        self.count += 1

        # Show progress
        logger.info(f"[{self.count}/{self.n_total}] Generating a report for community (ID:{community['community_id']}) with {len(direct_nodes)} direct nodes and {len(child_reports)} sub communities (IDs:{[c['community_id'] for c in child_reports]})...")

        # Generate a prompt
        prompt = self.generate_prompt(
            direct_nodes=direct_nodes,
            child_reports=child_reports,
        )

        # Generate a report based on the prompt
        generated_text = self.model.generate(prompt)

        # Process the generated report
        processed_title, processed_text = self.parse_generated_text_fn(generated_text)

        return {"title": processed_title, "text": processed_text} | community

    def generate_prompt(
        self,
        direct_nodes: list[str],
        child_reports: list[Passage]
    ) -> str:
        """Generate an LLM prompt from community data."""

        # Limit number of direct nodes
        if len(direct_nodes) >= 100:
            logger.info(f"[{self.count}/{self.n_total}] Reducing nodes to top 100 primary nodes among {len(direct_nodes)} nodes")
            direct_nodes = [
                n for n, _ in sorted(
                    self.graph.subgraph(direct_nodes).degree(),
                    key=lambda x: x[1],
                    reverse=True
                )[:100]
            ]

        # Gather edges among the nodes
        edges = [
            (h,t,p) for h,t,p in self.graph.edges(direct_nodes, data=True)
            if (h in direct_nodes) and (t in direct_nodes)
        ]

        assert len(direct_nodes) + len(edges) + len(child_reports) > 0

        content_prompt = ""

        # Generate prompt part for nodes
        if len(direct_nodes) > 0:
            content_prompt += "Entities:\n"
            for node in direct_nodes:
                props = self.graph.nodes[node]
                name = props["name"].replace("|", " ")
                etype = props["entity_type"].replace("|", " ")
                desc = props["description"].replace("|", " ").replace("\n", " ").rstrip()
                if desc == "":
                    desc = "N/A"
                content_prompt += f"- {name} | {etype} | {desc}\n"
            content_prompt += "\n"

        # Generate prompt part for edges
        if len(edges) > 0:
            content_prompt += "Relationships:\n"
            for head, tail, props in edges:
                head_name = self.graph.nodes[head]["name"].replace("|", " ")
                tail_name = self.graph.nodes[tail]["name"].replace("|", " ")
                relation = props["relation"]
                relation = self.relation_map.get(relation, relation).replace("|", " ")
                content_prompt += f"- {head_name} | {relation} | {tail_name}\n"
            content_prompt += "\n"

        # Generate prompt part for child reports
        if len(child_reports) > 0:
            content_prompt += "Sub-Communities' Reports:\n"
            content_prompt += "\n"
            for c_i, child_report in enumerate(child_reports):
                title = child_report["title"].strip()
                text = child_report["text"].strip()
                content_prompt += f"[Sub-Community {c_i+1}]\n"
                content_prompt += f"Title: {title}\n"
                content_prompt += f"{text}\n"
                if c_i < len(child_reports) - 1:
                    content_prompt += "\n"

        # Finalize the prompt
        prompt = self.prompt_template.format(
            content_prompt=content_prompt.strip()
        )

        return prompt


def parse_generated_text(generated_text: str) -> tuple[str, str]:
    """Parse the LLM output into (title, summary text)."""

    json_obj = utils.safe_json_loads(generated_text=generated_text, fallback=None)
    if json_obj is None:
        return "No Title", generated_text

    try:
        title = json_obj["title"]
        text = json_obj["summary"] + "\n"
        for i, finding in enumerate(json_obj["findings"]):
            text += f"[Finding {i+1}] {finding['summary']}: {finding['explanation']}\n"
        text = text.rstrip()
        return title, text
    except Exception as e:
        logger.warning(f"Failed to parse structured JSON: {e}")
        return "No Title", generated_text

