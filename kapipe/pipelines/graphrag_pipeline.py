from __future__ import annotations

import os
from typing import Any

import networkx as nx
from tqdm import tqdm

from .. import utils
from ..chunking.base import BaseChunker
from ..community_clustering.base import BaseCommunityClusterer
from ..datatypes import (
    CommunityRecord,
    ContextsForOneExample,
    Document,
    Passage,
    Question,
)
from ..docre.base import BaseDocRE
from ..ed_reranking.base import BaseEDReranker
from ..ed_retrieval.base import BaseEDRetriever
from ..entity_graph_construction.base import BaseEntityGraphConstructor
from ..ner.base import BaseNER
from ..passage_retrieval.base import BasePassageRetriever
from ..qa.base import BaseQA
from ..report_generation.base import BaseReportGenerator


class GraphRAGPipeline:
    """Pipeline for running the full GraphRAG workflow or one component."""

    def __init__(
        self,
        ner: BaseNER | None,
        ed_retrieval: BaseEDRetriever | None,
        ed_reranking: BaseEDReranker | None,
        docre: BaseDocRE | None,
        entity_graph_construction: BaseEntityGraphConstructor | None,
        community_clustering: BaseCommunityClusterer | None,
        report_generation: BaseReportGenerator | None,
        chunker: BaseChunker | None,
        passage_retrieval: BasePassageRetriever | None,
        qa: BaseQA | None,
    ) -> None:

        self.ner: BaseNER | None = ner
        self.ed_retrieval: BaseEDRetriever | None = ed_retrieval
        self.ed_reranking: BaseEDReranker | None = ed_reranking
        self.docre: BaseDocRE | None = docre
        self.entity_graph_construction: BaseEntityGraphConstructor | None = (
            entity_graph_construction
        )
        self.community_clustering: BaseCommunityClusterer | None = (
            community_clustering
        )
        self.report_generation: BaseReportGenerator | None = report_generation
        self.chunker: BaseChunker | None = chunker
        self.passage_retrieval: BasePassageRetriever | None = passage_retrieval
        self.qa: BaseQA | None = qa

    ####################
    # Indexing
    ####################

    def make_index(
        self,
        # Input
        documents: list[Document] | None,
        # Output directory
        index_dir: str,
        # Component-specific arguments
        retrieval_size: int,
        window_size: int,
        entity_dict_path: str | None = None,
        additional_triples_path: str | None = None,
        node_attr_keys: tuple[str, ...] = ("name", "entity_type", "description"),
        edge_attr_keys: tuple[str, ...] = ("relation",),
        passage_retrieval_indexing_kwargs: dict[str, Any] | None = None,
        # Target component for indexing
        target_component: str | None = None,
        input_artifact_paths: dict[str, str] | None = None,
    ) -> None:
        """Build the full index or run one selected indexing component."""

        # Use empty mappings when optional mappings are omitted
        if passage_retrieval_indexing_kwargs is None:
            passage_retrieval_indexing_kwargs = {}
        if input_artifact_paths is None:
            input_artifact_paths = {}

        # Validate the target component
        valid_target_components: list[str] = [
            "triple_extraction",
            "entity_graph_construction",
            "community_clustering",
            "report_generation",
            "chunking",
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
            "documents_with_triples",
            "graph",
            "communities",
            "reports",
            "chunked_reports",
        }
        unknown_artifact_names: set[str] = (
            set(input_artifact_paths) - valid_artifact_names
        )
        if unknown_artifact_names:
            raise ValueError(
                "Unknown indexing input artifacts: "
                f"{sorted(unknown_artifact_names)}."
            )

        # Create the common destination for every indexing artifact
        utils.mkdir(index_dir)

        ##############################
        # [Step 1] Triple Extraction
        ##############################

        # Extract relational triples from the input documents
        if target_component is None or target_component == "triple_extraction":
            # Validate that every Triple Extraction components are initialized
            if self.ner is None:
                raise ValueError("NER component is not initialized.")
            if self.ed_retrieval is None:
                raise ValueError("Entity retrieval component is not initialized.")
            if self.ed_reranking is None:
                raise ValueError("Entity reranking component is not initialized.")
            if self.docre is None:
                raise ValueError(
                    "Document-level relation extraction component is not initialized."
                )

            # Validate that documents are provided for triple extraction
            if documents is None:
                raise ValueError(
                    "Argument `documents` is required for triple extraction."
                )

            # Extract triples document by document
            documents_with_triples: list[Document] = []
            for document in tqdm(documents, desc="Extracting triples"):
                # Extract entity mentions
                document = self.ner.extract(document=document)

                # Retrieve candidate entities
                document, candidate_entities_for_doc = self.ed_retrieval.search(
                    document=document,
                    retrieval_size=retrieval_size,
                )

                # Rerank candidate entities
                document = self.ed_reranking.rerank(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                )

                # Extract document-level relations
                document = self.docre.extract(document=document)
                documents_with_triples.append(document)

            # Save documents with extracted triples
            utils.write_json(
                os.path.join(index_dir, "documents_with_triples.json"),
                documents_with_triples,
            )

        ######################################
        # [Step 2] Entity Graph Construction
        ######################################

        # Construct the entity graph
        if (
            target_component is None
            or target_component == "entity_graph_construction"
        ):
            # Validate that the Entity Graph Construction component is initialized
            if self.entity_graph_construction is None:
                raise ValueError(
                    "Entity graph construction component is not initialized."
                )

            # Select the path to the documents with extracted triples
            documents_with_triples_path: str = input_artifact_paths.get(
                "documents_with_triples",
                os.path.join(index_dir, "documents_with_triples.json"),
            )

            # Construct the entity graph from the extracted triples
            graph: nx.MultiDiGraph = (
                self.entity_graph_construction.construct_entity_graph(
                    documents_path_list=[documents_with_triples_path],
                    entity_dict_path=entity_dict_path,
                    additional_triples_path=additional_triples_path,
                )
            )

            # Save the entity graph
            nx.write_graphml(graph, os.path.join(index_dir, "graph.graphml"))

        #################################
        # [Step 3] Community Clustering
        #################################

        # Cluster the entity graph into communities
        if (
            target_component is None
            or target_component == "community_clustering"
        ):
            # Validate that the Community Clustering component is initialized
            if self.community_clustering is None:
                raise ValueError(
                    "Community clustering component is not initialized."
                )

            # Load the graph for standalone execution
            if target_component is not None:
                graph = nx.read_graphml(
                    input_artifact_paths.get(
                        "graph",
                        os.path.join(index_dir, "graph.graphml"),
                    )
                )

            # Cluster graph communities
            communities: list[CommunityRecord] = (
                self.community_clustering.cluster_communities(graph=graph)
            )

            # Save community records
            utils.write_json(
                os.path.join(index_dir, "communities.json"),
                communities,
            )

        ##############################
        # [Step 4] Report Generation
        ##############################

        # Generate textual reports from graph communities
        if target_component is None or target_component == "report_generation":
            # Validate that the Report Generation component is initialized
            if self.report_generation is None:
                raise ValueError("Report generation component is not initialized.")

            # Load the graph and communities for standalone execution
            if target_component is not None:
                graph = nx.read_graphml(
                    input_artifact_paths.get(
                        "graph",
                        os.path.join(index_dir, "graph.graphml"),
                    )
                )
                communities = utils.read_json(
                    input_artifact_paths.get(
                        "communities",
                        os.path.join(index_dir, "communities.json"),
                    )
                )

            # Generate community reports
            reports: list[Passage] = (
                self.report_generation.generate_community_reports(
                    graph=graph,
                    communities=communities,
                    node_attr_keys=node_attr_keys,
                    edge_attr_keys=edge_attr_keys,
                )
            )

            # Save community reports
            utils.write_jsonl(os.path.join(index_dir, "reports.jsonl"), reports)

        #####################
        # [Step 5] Chunking
        #####################

        # Split the community reports into retrieval units
        if target_component is None or target_component == "chunking":
            # Validate that the Chunking component is initialized
            if self.chunker is None:
                raise ValueError("Chunker is not initialized.")

            # Load community reports for standalone execution
            if target_component is not None:
                reports = utils.read_jsonl(
                    input_artifact_paths.get(
                        "reports",
                        os.path.join(index_dir, "reports.jsonl"),
                    )
                )

            # Split each report into smaller chunks
            chunked_reports: list[Passage] = []
            for report in tqdm(reports, desc="Chunking reports"):
                chunked_reports.extend(
                    self.chunker.split_passage_to_chunked_passages(
                        passage=report,
                        window_size=window_size,
                    )
                )

            # Save chunked reports
            utils.write_jsonl(
                os.path.join(index_dir, "chunked_reports.jsonl"),
                chunked_reports,
            )

        ########################################
        # [Step 6a] Passage Retrieval (Indexing)
        ########################################

        # Build the passage retrieval index
        if (
            target_component is None
            or target_component == "passage_retrieval_indexing"
        ):
            # Validate that the Passage Retrieval component is initialized
            if self.passage_retrieval is None:
                raise ValueError("Passage retrieval component is not initialized.")

            # Load chunked reports for standalone execution
            if target_component is not None:
                chunked_reports = utils.read_jsonl(
                    input_artifact_paths.get(
                        "chunked_reports",
                        os.path.join(index_dir, "chunked_reports.jsonl"),
                    )
                )

            # Build and save the component-specific retrieval index
            passage_retrieval_index_dir: str = os.path.join(
                index_dir,
                "passage_retrieval_index",
            )
            self.passage_retrieval.make_index(
                passages=chunked_reports,
                index_dir=passage_retrieval_index_dir,
                **passage_retrieval_indexing_kwargs,
            )

    ####################
    # Inference
    ####################

    def load_index(self, index_dir: str) -> None:
        """Load all indices required for inference."""
        self.passage_retrieval.load_index(
            index_dir=os.path.join(index_dir, "passage_retrieval_index"),
        )

    def infer(
        self,
        questions: list[Question],
        top_k: int,
    ) -> list[Question]:
        """Run all inference components and answer the questions."""

        # Validate that every inference component is initialized
        if self.passage_retrieval is None:
            raise ValueError("Passage retrieval component is not initialized.")
        if self.qa is None:
            raise ValueError("QA component is not initialized.")

        # Validate that top_k is a positive integer
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        results: list[Question] = []
        for question in tqdm(questions, desc="Answering questions"):

            ######################################
            # [Step 6b] Passage Retrieval (Search)
            ######################################

            # Retrieve relevant chunked reports
            retrieved_chunked_reports: list[Passage] = self.passage_retrieval.search(
                queries=[question["question"]],
                top_k=top_k,
            )[0]

            # Wrap retrieved chunked reports in the QA context format
            contexts_for_question: ContextsForOneExample = {
                "question_key": question["question_key"],
                "contexts": retrieved_chunked_reports,
            }

            ###############################
            # [Step 7] Answer Generation
            ###############################

            # Generate the final answer
            result: Question = self.qa.answer(
                question=question,
                contexts_for_question=contexts_for_question,
            )

            # Preserve retrieved contexts
            result["contexts"] = retrieved_chunked_reports

            results.append(result)

        return results
