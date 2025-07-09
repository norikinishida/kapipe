from __future__ import annotations

import logging

import networkx as nx
from tqdm import tqdm

from .. import utils
from ..datatypes import EntityPage


logger = logging.getLogger(__name__)


class GraphConstructor:

    def __init__(self):
        pass
    
    def construct_knowledge_graph(
        self,
        path_documents_list: list[str] | None,
        path_entity_dict: str,
        path_additional_triples: str | None = None,
        excluded_filenames: list[str] | None = None
    ) -> nx.MultiDiGraph:
        if excluded_filenames is None:
            excluded_filenames = []

        # Load the entity dictionary
        # Entity dictionary is a mapping from an entity ID (str) to the corresponding entity page
        entity_dict = utils.read_json(path_entity_dict)
        entity_dict = {epage["entity_id"]: epage for epage in entity_dict}

        # Initialize a directed, multi-edge graph
        graph = nx.MultiDiGraph()

        # Add triples from external file if provided
        if path_additional_triples is not None:
            # Load the additional triples
            triples = utils.read_json(path_additional_triples)
            for triple in tqdm(triples, desc="Adding additional triples"):
                # Add the single triple to the graph
                self._add_triple_to_graph(
                    graph=graph,
                    triple=triple,
                    entity_dict=entity_dict,
                    doc_key="ExistingKG"
                )

        # Add triples from documents
        if path_documents_list is None:
            path_documents_list = []
        for path_documents in path_documents_list:
            # Load the documents
            documents = utils.read_json(path_documents)
            logger.info(f"Loading triples from {len(documents)} documents in {path_documents}")
            for document in tqdm(documents, f"Processing {path_documents}"):
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

        # Final deduplication and doc_key_list consolidation
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
        triple: dict[str,str],
        entity_dict: dict[str, EntityPage],
        doc_key: str
    ) -> None:
        # Extract the elements
        head_id = triple["head"]
        tail_id = triple["tail"]
        relation = triple["relation"]

        # Get entity pages for the head/tail entities
        head_page = entity_dict.get(head_id, None)
        tail_page = entity_dict.get(tail_id, None)
        if head_page is None or tail_page is None:
            return

        # Get types for the head/tail entities
        head_type = triple.get("head_type") or self._infer_entity_type(epage=head_page)
        tail_type = triple.get("tail_type") or self._infer_entity_type(epage=tail_page)

        # Add the head and tail entities as nodes
        for entity_id, entity_page, entity_type in [
            (head_id, head_page, head_type),
            (tail_id, tail_page, tail_type)
        ]:
            if entity_id not in graph:
                graph.add_node(
                    entity_id,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=entity_page["canonical_name"],
                    description=entity_page["description"],
                    doc_key_list=[doc_key]
                )
            else:
                graph.nodes[entity_id]["doc_key_list"].append(doc_key)

        # Add the relation as an edge
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
        # Help to infer entity type from entity page
        if "entity_type_names" in epage:
            return " | ".join(epage["entity_type_names"])
        elif "entity_type" in epage:
            return epage["entity_type"]
        return "UNKNOWN"

