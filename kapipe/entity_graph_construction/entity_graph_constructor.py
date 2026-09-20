from __future__ import annotations

import logging
from typing import Any

import networkx as nx
from tqdm import tqdm

from ..datatypes import Document, EntityPage
from .base import BaseEntityGraphConstructor


logger = logging.getLogger(__name__)


class EntityGraphConstructor(BaseEntityGraphConstructor):

    def __init__(
        self,
        missing_entity_policy: str = "keep",
        missing_entity_description: str = "NO DESCRIPTION.",
    ):
        # Validate how triples with missing entity pages should be handled
        if missing_entity_policy not in ["keep", "drop"]:
            raise ValueError(
                f"Invalid missing_entity_policy: {missing_entity_policy}"
            )

        self.missing_entity_policy = missing_entity_policy
        self.missing_entity_description = missing_entity_description
    
    def construct_entity_graph(
        self,
        documents: list[Document] | None,
        entity_dict: list[EntityPage] | None,
        additional_triples: list[dict[str, Any]] | None,
    ) -> nx.MultiDiGraph:
        """Construct a directed, multi-edge graph from the provided documents and entity dictionary."""

        # Create a mapping from entity ID to entity page for quick lookup
        if entity_dict is not None:
            entity_dict: dict[str, EntityPage] = {
                epage["entity_id"]: epage
                for epage in entity_dict
            }
            logger.info(
                f"Received entity dictionary with {len(entity_dict)} entries."
            )
        else:
            entity_dict = {}
            logger.info(
                "No entity dictionary provided. "
                "Falling back to document entities only."
            )

        # Initialize a directed, multi-edge graph
        graph = nx.MultiDiGraph()

        # Add optional triples from an existing graph
        if additional_triples is not None:
            for triple in tqdm(
                additional_triples,
                desc="Adding additional triples"
            ):
                # Add the single triple to the graph
                self._add_triple_to_graph(
                    graph=graph,
                    triple=triple,
                    entity_dict=entity_dict,
                    doc_key="ExistingKG",
                )            

        # Add triples from documents
        if documents is None:
            documents = []
        for document in tqdm(documents, f"Processing documents"):
            # Get the associated triples from the document
            doc_key = document["doc_key"]
            triples = document["relations"]
            entities = document["entities"]
            for triple in triples:
                head_index = triple["arg1"]
                tail_index = triple["arg2"]
                relation = triple["relation"]
                head_id = entities[head_index]["entity_id"]
                tail_id = entities[tail_index]["entity_id"]
                head_type = entities[head_index]["entity_type"]
                tail_type = entities[tail_index]["entity_type"]
                triple_obj = {
                    "head": head_id,
                    "tail": tail_id,
                    "relation": relation,
                    "head_type": head_type,
                    "tail_type": tail_type
                }

                # Add the single triple to the graph
                self._add_triple_to_graph(
                    graph=graph,
                    triple=triple_obj,
                    entity_dict=entity_dict,
                    doc_key=doc_key
                )

        # Consolidate document keys for every node and edge
        for node, prop in graph.nodes(data=True):
            graph.nodes[node]["doc_key_list"] = "|".join(
                sorted(list(set(prop["doc_key_list"])))
            )
        for h, t, k, prop in graph.edges(keys=True, data=True):
            graph.edges[h, t, k]["doc_key_list"] = "|".join(
                sorted(list(set(prop["doc_key_list"])))
            )

        logger.info(f"The number of nodes: {graph.number_of_nodes()}") 
        logger.info(f"The number of edges: {graph.number_of_edges()}") 

        return graph

    def _add_triple_to_graph(
        self,
        graph: nx.MultiDiGraph,
        triple: dict[str, Any],
        entity_dict: dict[str, EntityPage],
        doc_key: str
    ) -> None:
        """Add a single triple to the graph, including its head and tail entities as nodes and the relation as an edge."""

        # Extract the elements
        head_id = triple["head"]
        tail_id = triple["tail"]
        relation = triple["relation"]

        # Get entity pages for the head/tail entities
        head_page = entity_dict.get(head_id, None)
        tail_page = entity_dict.get(tail_id, None)

        # Drop the triple when missing entity pages are not allowed
        if self.missing_entity_policy == "drop":
            if head_page is None or tail_page is None:
                return

        # Create fallback entity pages for missing head/tail entities
        if head_page is None:
            head_page = {
                "entity_id": head_id,
                "canonical_name": head_id, # mention name (normalized)
                "description": self.missing_entity_description,
            }
        if tail_page is None:
            tail_page = {
                "entity_id": tail_id,
                "canonical_name": tail_id,
                "description": self.missing_entity_description,
            }

        # Use the explicit types when the triple provides it.
        # Otherwise, infer the entity type from the entity page.
        head_type = triple.get("head_type") or self._infer_entity_type(epage=head_page)
        tail_type = triple.get("tail_type") or self._infer_entity_type(epage=tail_page)

        # Add the head and tail entities as nodes
        for entity_id, entity_page, entity_type in [
            (head_id, head_page, head_type),
            (tail_id, tail_page, tail_type)
        ]:
            # Add a new node when the entity is unseen
            if entity_id not in graph:
                graph.add_node(
                    entity_id,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=entity_page["canonical_name"],
                    description=entity_page["description"],
                    doc_key_list=[doc_key]
                )
            # Append provenance when the entity already exists
            else:
                graph.nodes[entity_id]["doc_key_list"].append(doc_key)

        # Add a new labeled edge when the relation is unseen for this node pair
        if not graph.has_edge(head_id, tail_id, relation):
            graph.add_edge(
                head_id,
                tail_id,
                key=relation,
                relation=relation,
                doc_key_list=[doc_key]
            )
        else:
            graph.edges[head_id, tail_id, relation]["doc_key_list"].append(doc_key)

    def _infer_entity_type(self, epage: EntityPage) -> str:
        """Infer the entity type from the entity page, if available."""

        # Help to infer entity type from entity page
        if "entity_type_names" in epage:
            return " | ".join(epage["entity_type_names"])
        elif "entity_type" in epage:
            return epage["entity_type"]

        return "UNKNOWN"

