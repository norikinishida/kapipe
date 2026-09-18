from __future__ import annotations

import copy
import logging
import os
from typing import Any

import networkx as nx
from tqdm import tqdm

from .. import utils
from ..context_formatting.base import BaseContextFormatter
from ..datatypes import ContextsForOneExample, Passage, Question
from ..graph_retrieval.base import BaseGraphRetriever
from ..passage_graph_construction.base import BasePassageGraphConstructor
from ..passage_retrieval.base import BasePassageRetriever
from ..proposition_extraction.base import BasePropositionExtractor
from ..proposition_relation_extraction.base import BasePropositionRelationExtractor
from ..proposition_relation_refinement.base import BasePropositionRelationRefiner
from ..qa.base import BaseQA


logger: logging.Logger = logging.getLogger(__name__)


class ProStructRAGPipeline:
    """Pipeline for running the full ProStruct-RAG workflow or one component."""

    def __init__(
        self,
        proposition_extraction: BasePropositionExtractor | None,
        proposition_relation_extraction: BasePropositionRelationExtractor | None,
        proposition_relation_refinement: BasePropositionRelationRefiner | None,
        passage_graph_construction: BasePassageGraphConstructor | None,
        passage_retrieval: BasePassageRetriever | None,
        graph_retrieval: BaseGraphRetriever | None,
        context_formatting: BaseContextFormatter | None,
        qa: BaseQA | None,
    ) -> None:

        self.proposition_extraction: BasePropositionExtractor | None = (
            proposition_extraction
        )
        self.proposition_relation_extraction: (
            BasePropositionRelationExtractor | None
        ) = proposition_relation_extraction
        self.proposition_relation_refinement: (
            BasePropositionRelationRefiner | None
        ) = proposition_relation_refinement
        self.passage_graph_construction: BasePassageGraphConstructor | None = (
            passage_graph_construction
        )
        self.passage_retrieval: BasePassageRetriever | None = passage_retrieval
        self.graph_retrieval: BaseGraphRetriever | None = graph_retrieval
        self.context_formatting: BaseContextFormatter | None = context_formatting
        self.qa: BaseQA | None = qa

    ####################
    # Indexing
    ####################

    def make_index(
        self,
        # Input
        passages: list[Passage] | None,
        # Output directory
        index_dir: str,
        # Component-specific arguments
        top_k: int,
        prefilter_k: int,
        search_batch_size: int,
        proposition_relation_extraction_indexing_kwargs: dict[str, Any] | None = None,
        passage_retrieval_indexing_kwargs: dict[str, Any] | None = None,
        # Target component for indexing
        target_component: str | None = None,
        input_artifact_paths: dict[str, str] | None = None,
        # Batch API
        batch_mode: str | None = None,
        batch_dir: str | None = None,
    ) -> None:
        """Build the full index or run one selected indexing component."""

        # Use empty mappings when optional mappings are omitted
        if proposition_relation_extraction_indexing_kwargs is None:
            proposition_relation_extraction_indexing_kwargs = {}
        if passage_retrieval_indexing_kwargs is None:
            passage_retrieval_indexing_kwargs = {}
        if input_artifact_paths is None:
            input_artifact_paths = {}

        # Validate the target component
        valid_target_components: list[str] = [
            "proposition_extraction",
            "proposition_relation_extraction",
            "proposition_relation_refinement",
            "passage_graph_construction",
            "passage_retrieval_indexing",
        ]
        if target_component is not None:
            if target_component not in valid_target_components:
                raise ValueError(
                    f"Unknown indexing target_component: {target_component}. "
                    f"Expected one of: {valid_target_components}."
                )

        # Validate that input artifacts are used only for standalone execution
        if input_artifact_paths:
            if target_component is None:
                raise ValueError(
                    "`input_artifact_paths` requires `target_component`."
                )

        # Validate input artifact names
        valid_artifact_names: set[str] = {
            "propositions",
            "triples",
            "refined_triples",
        }
        unknown_artifact_names: set[str] = (
            set(input_artifact_paths) - valid_artifact_names
        )
        if unknown_artifact_names:
            raise ValueError(
                "Unknown indexing input artifacts: "
                f"{sorted(unknown_artifact_names)}."
            )

        # Validate that a target component is specified when using batch mode
        if batch_mode is not None:
            if target_component is None:
                raise ValueError(
                    "`target_component` is required when `batch_mode` is specified."
                )

        # Create the common destination for every indexing artifact
        utils.mkdir(index_dir)

        #################################
        # [Step 1] Proposition Extraction
        #################################

        # Extract propositions
        if (
            target_component is None
            or target_component == "proposition_extraction"
        ):
            # Validate that the Proposition Extraction component is initialized
            if self.proposition_extraction is None:
                raise ValueError(
                    "Proposition Extraction component is not initialized."
                )

            # Validate that the input passages are provided
            if passages is None:
                raise ValueError(
                    "Argument `passages` is required for proposition extraction."
                )

            # Apply the Proposition Extraction component to the passages
            if batch_mode is None:
                propositions: list[Passage] = []
                for passage in tqdm(passages, desc="Extracting propositions"):
                    propositions_for_passage: list[Passage] = (
                        self.proposition_extraction.extract(
                            passage=passage,
                        )
                    )
                    propositions.extend(propositions_for_passage)

                logger.info(
                    f"Extracted {len(propositions)} propositions "
                    f"from {len(passages)} passages"
                )

                # Save the Proposition Extraction results
                utils.write_jsonl(
                    os.path.join(index_dir, "propositions.jsonl"),
                    propositions,
                )

            elif batch_mode == "submit":
                # Submit prompts 
                batch_ids: list[str] = self.proposition_extraction.submit_batch(
                    passages=passages
                )
                utils.mkdir(batch_dir)
                utils.write_json(
                    os.path.join(batch_dir, "batch_ids.json"),
                    batch_ids,
                )
                logger.info(f"Submitted batches {batch_ids}")

            elif batch_mode == "fetch":
                # Fetch and process the responses
                batch_ids: list[str] = utils.read_json(
                    os.path.join(batch_dir, "batch_ids.json")
                )
                propositions: list[Passage] = (
                    self.proposition_extraction.fetch_and_process_batch(
                        passages=passages,
                        batch_ids=batch_ids,
                    )
                )

                logger.info(
                    f"Extracted {len(propositions)} propositions "
                    f"from {len(passages)} passages"
                )

                # Save the Proposition Extraction results
                utils.write_jsonl(
                    os.path.join(index_dir, "propositions.jsonl"),
                    propositions,
                )

            else:
                raise ValueError(
                    f"Invalid batch_mode: {batch_mode}. "
                    "Expected None, 'submit', or 'fetch'."
                )

        #################################
        # [Step 2] Proposition Relation Extraction
        #################################

        # Extract proposition relations
        if (
            target_component is None
            or target_component == "proposition_relation_extraction"
        ):
            # Validate that the Proposition Relation Extraction component 
            # is initialized.
            if self.proposition_relation_extraction is None:
                raise ValueError(
                    "Proposition relation extraction component is not initialized."
                )

            # Load propositions for standalone execution
            if target_component is not None:
                propositions: list[Passage] = utils.read_jsonl(
                    input_artifact_paths.get(
                        "propositions",
                        os.path.join(index_dir, "propositions.jsonl"),
                    )
                )

            # Build index for propositions
            intermediate_index_dir: str = os.path.join(
                index_dir,
                "intermediate_passage_retrieval_index",
            )
            self.proposition_relation_extraction.make_index(
                propositions=propositions,
                index_dir=intermediate_index_dir,
                **proposition_relation_extraction_indexing_kwargs,
            )

            # Retrieve tail propositions for each proposition
            batch_tail_propositions: list[list[Passage]] = (
                self.proposition_relation_extraction.batch_retrieve_tail_propositions(
                    head_propositions=propositions,
                    top_k=top_k,
                    prefilter_k=prefilter_k,
                    batch_size=search_batch_size,
                )
            )

            # Extract proposition relations for each proposition 
            if batch_mode is None:
                triples: list[dict[str, Any]] = []
                for head_proposition, tail_propositions in tqdm(
                    zip(propositions, batch_tail_propositions),
                    total=len(propositions),
                    desc="Extracting proposition relations",
                ):
                    triples_for_head: list[dict[str, Any]] = (
                        self.proposition_relation_extraction.extract(
                            head_proposition=head_proposition,
                            tail_propositions=tail_propositions,
                        )
                    )
                    triples.extend(triples_for_head)

                logger.info(
                    f"Extracted {len(triples)} triples from "
                    f"{len(propositions)} propositions"
                )

                # Save the Proposition Relation Extraction results 
                output_triples_path = os.path.join(index_dir, "triples.json")
                utils.write_json(output_triples_path, triples)

            elif batch_mode == "submit":
                # Submit prompts
                batch_ids: list[str] = (
                    self.proposition_relation_extraction.submit_batch(
                        head_propositions=propositions,
                        batch_tail_propositions=batch_tail_propositions,
                    )
                )
                utils.mkdir(batch_dir)
                utils.write_json(
                    os.path.join(batch_dir, "batch_ids.json"),
                    batch_ids,
                )
                logger.info(f"Submitted batches {batch_ids}")

            elif batch_mode == "fetch":
                # Fetch and process the responses
                batch_ids: list[str] = utils.read_json(
                    os.path.join(batch_dir, "batch_ids.json")
                )
                triples: list[dict[str, Any]] = (
                    self.proposition_relation_extraction.fetch_and_process_batch(
                        head_propositions=propositions,
                        batch_tail_propositions=batch_tail_propositions,
                        batch_ids=batch_ids,
                    )
                )

                logger.info(
                    f"Extracted {len(triples)} triples from "
                    f"{len(propositions)} propositions"
                )

                # Save the Proposition Relation Extraction results 
                output_triples_path = os.path.join(index_dir, "triples.json")
                utils.write_json(output_triples_path, triples)

            else:
                raise ValueError(
                    f"Invalid batch_mode: {batch_mode}. "
                    "Expected None, 'submit', or 'fetch'"
                )

        #################################
        # [Step 3] Proposition Relation Refinement
        #################################

        # Refine proposition relations
        if (
            target_component is None
            or target_component == "proposition_relation_refinement"
        ):
            # Validate that the Proposition Relation Refinement component 
            # is initialized.
            if self.proposition_relation_refinement is None:
                raise ValueError(
                    "Proposition relation refinement component is not initialized."
                )

            # Load triples for standalone execution
            if target_component is not None:
                triples: list[dict[str, Any]] = utils.read_json(
                    input_artifact_paths.get(
                        "triples",
                        os.path.join(index_dir, "triples.json"),
                    )
                )

            # Apply the Proposition Relation Refinement component to the triples
            if batch_mode is None:
                refined_triples: list[dict[str, Any]] = []
                n_deleted: int = 0
                for triple in tqdm(
                    triples,
                    desc="Refining proposition relations",
                ):
                    refined_triple: dict[str, Any] = (
                        self.proposition_relation_refinement.refine(
                            triple=triple,
                        )
                    )
                    # Remove triples classified as NOREL
                    if refined_triple["relation"] == "NOREL":
                        n_deleted += 1
                        continue
                    refined_triples.append(refined_triple)

                logger.info(
                    f"Refinement complete: {len(refined_triples)} triples kept, "
                    f"{n_deleted} triples removed"
                )

                # Save the Proposition Relation Refinement results
                utils.write_json(
                    os.path.join(index_dir, "refined_triples.json"),
                    refined_triples,
                )

            elif batch_mode == "submit":
                # Submit prompts
                batch_ids: list[str] = (
                    self.proposition_relation_refinement.submit_batch(
                        triples=triples,
                    )
                )
                utils.mkdir(batch_dir)
                utils.write_json(
                    os.path.join(batch_dir, "batch_ids.json"),
                    batch_ids,
                )
                logger.info(f"Submitted batches {batch_ids}")

            elif batch_mode == "fetch":
                # Fetch and process the responses
                batch_ids: list[str] = utils.read_json(
                    os.path.join(batch_dir, "batch_ids.json")
                )
                tmp_refined_triples: list[dict[str, Any]] = (
                    self.proposition_relation_refinement.fetch_and_process_batch(
                        triples=triples,
                        batch_ids=batch_ids,
                    )
                )

                # Remove triples classified as NOREL
                refined_triples: list[dict[str, Any]] = []
                n_deleted: int = 0
                for refined_triple in tmp_refined_triples:
                    if refined_triple["relation"] == "NOREL":
                        n_deleted += 1
                        continue
                    refined_triples.append(refined_triple)

                logger.info(
                    f"Refinement complete: {len(refined_triples)} triples kept, "
                    f"{n_deleted} triples removed"
                )

                # Save the Proposition Relation Refinement results
                utils.write_json(
                    os.path.join(index_dir, "refined_triples.json"),
                    refined_triples,
                )

            else:
                raise ValueError(
                    f"Invalid batch_mode: {batch_mode}. "
                    "Expected None, 'submit', or 'fetch'."
                )

        #################################
        # [Step 4] Passage Graph Construction
        #################################

        # Construct the passage graph
        if (
            target_component is None
            or target_component == "passage_graph_construction"
        ):
            # Validate that the Passage Graph Construction component is initialized
            if self.passage_graph_construction is None:
                raise ValueError(
                    "Passage graph construction component is not initialized."
                )

            # Load propositions and refined triples for standalone execution
            if target_component is not None:
                propositions: list[Passage] = utils.read_jsonl(
                    input_artifact_paths.get(
                        "propositions",
                        os.path.join(index_dir, "propositions.jsonl"),
                    )
                )
                refined_triples: list[dict[str, Any]] = utils.read_json(
                    input_artifact_paths.get(
                        "refined_triples",
                        os.path.join(index_dir, "refined_triples.json"),
                    )
                )

            # Apply the Passage Graph Construction component to the triples
            graph: nx.DiGraph = (
                self.passage_graph_construction.construct_passage_graph(
                    passages=propositions,
                    triples=refined_triples,
                )
            )

            # Show statistics
            _show_graph_statistics(graph)

            # Save the Passage Graph Construction results
            nx.write_graphml(
                graph,
                os.path.join(index_dir, "graph.graphml"),
            )

        #################################
        # [Step 5a] Passage Retrieval (Indexing)
        #################################

        # Build the passage retrieval index when the component is selected
        if (
            target_component is None
            or target_component == "passage_retrieval_indexing"
        ):
            # Validate that the Passage Retrieval component is initialized
            if self.passage_retrieval is None:
                raise ValueError(
                    "Passage retrieval component is not initialized."
                )

            # Load propositions for standalone execution
            if target_component is not None:
                propositions: list[Passage] = utils.read_jsonl(
                    input_artifact_paths.get(
                        "propositions",
                        os.path.join(index_dir, "propositions.jsonl"),
                    )
                )

            # Build index
            passage_retrieval_index_dir: str = os.path.join(
                index_dir,
                "passage_retrieval_index",
            )
            self.passage_retrieval.make_index(
                passages=propositions,
                index_dir=passage_retrieval_index_dir,
                **passage_retrieval_indexing_kwargs,
            )

    ####################
    # Inference
    ####################

    def load_index(
        self,
        index_dir: str,
    ) -> None:
        """Load all necessary indices for inference."""
        self.passage_retrieval.load_index(
            index_dir=os.path.join(index_dir, "passage_retrieval_index")
        )
        graph: nx.DiGraph = nx.read_graphml(
            os.path.join(index_dir, "graph.graphml")
        )
        self.graph_retrieval.make_index(graph=graph)

    def infer(
        self,
        # Input
        questions: list[Question],
        # Component-specific parameters
        top_k: int,
        hop_size: int,
        # ProStruct-RAG-specific parameters
        remove_same_timestamp_updates: bool = True,
        append_question_timestamp: bool = True,
        # Batch API
        batch_mode: str | None = None,
        batch_dir: str | None = None,
    ) -> list[Question] | None:
        """Run all inference components and answer one question."""

        # Validate that every inference component is initialized
        if self.passage_retrieval is None:
            raise ValueError("Passage retrieval component is not initialized.")
        if self.graph_retrieval is None:
            raise ValueError("Graph retrieval component is not initialized.")
        if self.context_formatting is None:
            raise ValueError("Context formatting component is not initialized.")
        if self.qa is None:
            raise ValueError("QA component is not initialized.")

        # Process each question individually
        question_with_time_list: list[Question] = []
        anchor_contexts_list: list[ContextsForOneExample] = []
        graph_contexts_list: list[ContextsForOneExample] = []
        formatted_contexts_list: list[ContextsForOneExample] = []
        for question in tqdm(questions, desc="Answering questions"):

            #################################
            # [Step 5b] Passage Retrieval (Search)
            #################################

            # Search top-k anchor propositions for the question
            anchor_propositions: list[Passage] = self.passage_retrieval.search(
                queries=[question["question"]],
                top_k=top_k,
            )[0]

            # Create a ContextsForOneExample object for the question
            anchor_contexts: ContextsForOneExample = {
                "question_key": question["question_key"],
                "contexts": anchor_propositions,
            }

            anchor_contexts_list.append(anchor_contexts)

            #################################
            # [Step 6] Graph Retrieval
            #################################

            # Extract the anchor node IDs for the current question
            anchor_node_ids: list[str] = [
                prop["passage_key"]
                for prop in anchor_contexts["contexts"]
            ]

            # Retrieve neighboring nodes and edges based on the anchor propositions
            nodes, edges = self.graph_retrieval.search(
                anchor_node_ids=anchor_node_ids,
                hop_size=hop_size,
            )

            # Represent the retrieved subgraph as contexts for the question
            graph_contexts: ContextsForOneExample = {
                "question_key": question["question_key"],
                "contexts": None,
                "nodes": nodes,
                "edges": edges,
            }

            graph_contexts_list.append(graph_contexts)

            #################################
            # [Step 7] Context Formatting
            #################################

            # Remove same-timestamp update edges when requested
            # TODO: Consider whether this filtering should be done at this stage
            filtered_edges: list[dict[str, Any]] = []
            for edge in edges:
                if remove_same_timestamp_updates:
                    relation: str = edge["relation"].lower()
                    if relation == "updates":
                        head_timestamp: str = nodes[edge["head"]]["timestamp"]
                        tail_timestamp: str = nodes[edge["tail"]]["timestamp"]
                        if head_timestamp == tail_timestamp:
                            continue
                filtered_edges.append(edge)

            # Convert the graph into LLM-readable context
            text: str = self.context_formatting.convert(
                nodes=nodes,
                edges=filtered_edges,
            )

            # Store the formatted context in the ContextsForOneExample format
            formatted_contexts: ContextsForOneExample = copy.deepcopy(graph_contexts)
            formatted_contexts["contexts"] = [
                {
                    "passage_key": f"{question['question_key']}/context#0000",
                    "text": text,
                }
            ]

            formatted_contexts_list.append(formatted_contexts)

            #################################
            # [Step 8-1] Answer Generation
            #################################

            # Add the query date using the ProStruct-RAG representation
            # TODO: Consider whether this timestamp appending should be done at this stage
            question_with_time: Question = copy.deepcopy(question)
            if append_question_timestamp:
                question_text: str = question_with_time["question"].strip()
                timestamp: str = question_with_time["timestamp"].strip()
                question_with_time["question"] = (
                    f"{question_text} (Date: {timestamp})"
                )

            question_with_time_list.append(question_with_time)

        #################################
        # [Step 8-2] Answer Generation
        #################################

        if batch_mode is None:
            results: list[Question] = []
            for (
                question_with_time,
                anchor_contexts,
                graph_contexts,
                formatted_contexts
            ) in zip(
                question_with_time_list,
                anchor_contexts_list,
                graph_contexts_list,
                formatted_contexts_list
            ):
                # Generate the final answer
                result: Question = self.qa.answer(
                    question=question_with_time,
                    contexts_for_question=formatted_contexts,
                )

                # Preserve all intermediate results in the question object
                result["anchor_contexts"] = anchor_contexts
                result["graph_contexts"] = graph_contexts
                result["formatted_contexts"] = formatted_contexts

                results.append(result)

        elif batch_mode == "submit":
            # Submit prompts
            batch_ids: list[str] = self.qa.submit_batch(
                questions=question_with_time_list,
                contexts=formatted_contexts_list,
            )
            utils.mkdir(batch_dir)
            utils.write_json(
                os.path.join(batch_dir, "batch_ids.json"),
                batch_ids
            )
            logger.info(f"Submitted batches {batch_ids}")

        elif batch_mode == "fetch":
            # Fetch and process the responses
            batch_ids: list[str] = utils.read_json(
                os.path.join(batch_dir, "batch_ids.json")
            )
            results: list[Question] = self.qa.fetch_and_process_batch(
                questions=question_with_time_list,
                contexts=formatted_contexts_list,
                batch_ids=batch_ids
            )

            for (
                result,
                anchor_contexts,
                graph_contexts,
                formatted_contexts
            ) in zip(
                results,
                anchor_contexts_list,
                graph_contexts_list,
                formatted_contexts_list
            ):
                # Preserve all intermediate results in the question object
                result["anchor_contexts"] = anchor_contexts
                result["graph_contexts"] = graph_contexts
                result["formatted_contexts"] = formatted_contexts
        else:
            raise ValueError(
                f"Invalid batch_mode: {batch_mode}. "
                "Expected None, 'submit', or 'fetch'."
            )

        if batch_mode == "submit":
            return None

        return results


def _show_graph_statistics(graph: nx.DiGraph) -> None:
    # Count nodes and edges
    num_nodes: int = graph.number_of_nodes()
    num_edges: int = graph.number_of_edges()

    # Calculate graph density
    density: float = nx.density(graph)

    # Calculate in-degree and out-degree statistics
    in_degrees: dict[Any, int] = dict(graph.in_degree())
    out_degrees: dict[Any, int] = dict(graph.out_degree())
    avg_in: float = sum(in_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_out: float = sum(out_degrees.values()) / num_nodes if num_nodes > 0 else 0
    max_in: int = max(in_degrees.values()) if num_nodes > 0 else 0
    max_out: int = max(out_degrees.values()) if num_nodes > 0 else 0

    # Count strongly and weakly connected components
    num_scc: int = nx.number_strongly_connected_components(graph)
    num_wcc: int = nx.number_weakly_connected_components(graph)

    # Organize graph statistics
    statistics: dict[str, Any] = {
        "nodes": num_nodes,
        "edges": num_edges,
        "density": density,
        "directed": graph.is_directed(),
        "avg_in_degree": avg_in,
        "avg_out_degree": avg_out,
        "max_in_degree": max_in,
        "max_out_degree": max_out,
        "num_strongly_connected_components": num_scc,
        "num_weakly_connected_components": num_wcc,
    }

    # Show every graph statistic
    for key, value in statistics.items():
        logger.info(f"{key}: {value}")
