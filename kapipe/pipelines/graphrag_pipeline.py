from __future__ import annotations

import logging
import os
from typing import Any

import networkx as nx
from tqdm import tqdm

from .. import utils
from ..chunking.base import BaseChunker
from ..community_clustering.base import BaseCommunityClusterer
from ..datatypes import (
    CandidateEntitiesForDocument,
    CommunityRecord,
    ContextsForOneExample,
    Document,
    EntityPage,
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


logger: logging.Logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Pipeline for running the full GraphRAG workflow or one component."""

    def __init__(
        self,
        ner: BaseNER,
        ed_retrieval: BaseEDRetriever,
        ed_reranking: BaseEDReranker,
        docre: BaseDocRE,
        entity_graph_construction: BaseEntityGraphConstructor,
        community_clustering: BaseCommunityClusterer,
        report_generation: BaseReportGenerator,
        chunker: BaseChunker,
        passage_retrieval: BasePassageRetriever,
        qa: BaseQA,
    ) -> None:

        self.ner: BaseNER = ner
        self.ed_retrieval: BaseEDRetriever = ed_retrieval
        self.ed_reranking: BaseEDReranker = ed_reranking
        self.docre: BaseDocRE = docre
        self.entity_graph_construction: BaseEntityGraphConstructor = (
            entity_graph_construction
        )
        self.community_clustering: BaseCommunityClusterer = (
            community_clustering
        )
        self.report_generation: BaseReportGenerator = report_generation
        self.chunker: BaseChunker = chunker
        self.passage_retrieval: BasePassageRetriever = passage_retrieval
        self.qa: BaseQA = qa

    ####################
    # Indexing
    ####################

    def make_index(
        self,
        # Input
        documents: list[Document],
        # Output directory
        index_dir: str,
        # Component-specific arguments
        retrieval_size: int,
        window_size: int,
        entity_dict: list[EntityPage] | None = None,
        additional_triples: list[dict[str, Any]] | None = None,
        node_attr_keys: tuple[str, ...] = ("name", "entity_type", "description"),
        edge_attr_keys: tuple[str, ...] = ("relation",),
        passage_retrieval_indexing_kwargs: dict[str, Any] | None = None,
        # Target component
        target_component: str | None = None,
        # Batch API
        batch_mode: str | None = None,
        batch_dir: str | None = None,
    ) -> None:
        """Build the full index or run one selected indexing component."""

        # Use empty mappings when optional mappings are omitted
        if passage_retrieval_indexing_kwargs is None:
            passage_retrieval_indexing_kwargs = {}

        # Validate the target component
        valid_target_components: list[str] = [
            "ner",
            "ed_retrieval",
            "ed_reranking",
            "docre",
            "entity_graph_construction",
            "community_clustering",
            "report_generation",
            "chunking",
            "passage_retrieval_indexing",
        ]
        if target_component is not None:
            if target_component not in valid_target_components:
                raise ValueError(
                    f"Unknown target_component: {target_component}. "
                    f"Expected one of: {valid_target_components}."
                )

        # Validate that a target component is specified when using batch mode
        if batch_mode is not None:
            if target_component is None:
                raise ValueError(
                    "`target_component` is required when `batch_mode` is specified."
                )

        # Create the common destination for every indexing artifact
        utils.mkdir(index_dir)

        ##############################
        # [Step 1a] NER
        ##############################

        # Extract entity mentions from the input documents
        if target_component is None or target_component == "ner":
            # Apply the NER component to the documents
            if batch_mode is None:
                documents_with_mentions: list[Document] = []
                for document in tqdm(documents, desc="Extracting entity mentions"):
                    documents_with_mentions.append(
                        self.ner.extract(document=document)
                    )

            elif batch_mode == "submit":
                # Submit prompts
                batch_ids: list[str] = self.ner.submit_batch(documents=documents)
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
                documents_with_mentions: list[Document] = (
                    self.ner.fetch_and_process_batch(
                        documents=documents,
                        batch_ids=batch_ids,
                    )
                )

            else:
                raise ValueError(
                    f"Invalid batch_mode: {batch_mode}. "
                    "Expected None, 'submit', or 'fetch'."
                )

            if batch_mode != "submit":
                # Save documents with extracted mentions
                utils.write_json(
                    os.path.join(index_dir, "documents_with_mentions.json"),
                    documents_with_mentions,
                )

        ##############################
        # [Step 1b] ED-Retrieval
        ##############################

        # Retrieve candidate entities for the extracted mentions
        if target_component is None or target_component == "ed_retrieval":
            # Load inputs for standalone execution
            if target_component is not None:
                documents_with_mentions: list[Document] = utils.read_json(
                    os.path.join(index_dir, "documents_with_mentions.json")
                )

            # Retrieve candidate entities document by document
            documents_with_candidates: list[Document] = []
            candidate_entities: list[CandidateEntitiesForDocument] = []
            for document in tqdm(
                documents_with_mentions, desc="Retrieving candidate entities"
            ):
                document_with_candidates, candidate_entities_for_doc = (
                    self.ed_retrieval.search(
                        document=document,
                        retrieval_size=retrieval_size,
                    )
                )
                documents_with_candidates.append(document_with_candidates)
                candidate_entities.append(candidate_entities_for_doc)

            # Save documents and candidate entities for reranking
            utils.write_json(
                os.path.join(index_dir, "documents_with_candidates.json"),
                documents_with_candidates,
            )
            utils.write_json(
                os.path.join(index_dir, "candidate_entities.json"),
                candidate_entities,
            )

        ##############################
        # [Step 1c] ED-Reranking
        ##############################

        # Rerank candidate entities for the extracted mentions
        if target_component is None or target_component == "ed_reranking":
            # Load inputs for standalone execution
            if target_component is not None:
                documents_with_candidates: list[Document] = utils.read_json(
                    os.path.join(index_dir, "documents_with_candidates.json")
                )
                candidate_entities: list[CandidateEntitiesForDocument] = (
                    utils.read_json(
                        os.path.join(index_dir, "candidate_entities.json")
                    )
                )

            # Apply the ED-Reranking component to the documents
            if batch_mode is None:
                documents_with_entities: list[Document] = []
                for document, candidate_entities_for_doc in tqdm(
                    zip(documents_with_candidates, candidate_entities),
                    total=len(documents_with_candidates),
                    desc="Reranking candidate entities",
                ):
                    documents_with_entities.append(
                        self.ed_reranking.rerank(
                            document=document,
                            candidate_entities_for_doc=candidate_entities_for_doc,
                        )
                    )

            elif batch_mode == "submit":
                # Submit prompts
                batch_ids: list[str] = self.ed_reranking.submit_batch(
                    documents=documents_with_candidates,
                    candidate_entities=candidate_entities,
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
                documents_with_entities: list[Document] = (
                    self.ed_reranking.fetch_and_process_batch(
                        documents=documents_with_candidates,
                        candidate_entities=candidate_entities,
                        batch_ids=batch_ids,
                    )
                )

            else:
                raise ValueError(
                    f"Invalid batch_mode: {batch_mode}. "
                    "Expected None, 'submit', or 'fetch'."
                )

            if batch_mode != "submit":
                # Save documents with disambiguated entities
                utils.write_json(
                    os.path.join(index_dir, "documents_with_entities.json"),
                    documents_with_entities,
                )

        ##############################
        # [Step 1d] DocRE
        ##############################

        # Extract document-level relations
        if target_component is None or target_component == "docre":
            # Load inputs for standalone execution
            if target_component is not None:
                documents_with_entities: list[Document] = utils.read_json(
                    os.path.join(index_dir, "documents_with_entities.json")
                )

            # Apply the DocRE component to the documents
            if batch_mode is None:
                documents_with_triples: list[Document] = []
                for document in tqdm(
                    documents_with_entities, desc="Extracting triples"
                ):
                    documents_with_triples.append(
                        self.docre.extract(document=document)
                    )

            elif batch_mode == "submit":
                # Submit prompts
                batch_ids: list[str] = self.docre.submit_batch(
                    documents=documents_with_entities
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
                documents_with_triples: list[Document] = (
                    self.docre.fetch_and_process_batch(
                        documents=documents_with_entities,
                        batch_ids=batch_ids,
                    )
                )

            else:
                raise ValueError(
                    f"Invalid batch_mode: {batch_mode}. "
                    "Expected None, 'submit', or 'fetch'."
                )

            if batch_mode != "submit":
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
            # Load inputs for standalone graph construction
            if target_component is not None:
                documents_with_triples: list[Document] = utils.read_json(
                    os.path.join(index_dir, "documents_with_triples.json")
                )

            # Construct the entity graph from the extracted triples
            graph: nx.MultiDiGraph = (
                self.entity_graph_construction.construct_entity_graph(
                    documents=documents_with_triples,
                    entity_dict=entity_dict,
                    additional_triples=additional_triples,
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
            # Load inputs for standalone execution
            if target_component is not None:
                graph = nx.read_graphml(
                    os.path.join(index_dir, "graph.graphml")
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
            # Load inputs for standalone execution
            if target_component is not None:
                graph = nx.read_graphml(
                    os.path.join(index_dir, "graph.graphml")
                )
                communities = utils.read_json(
                    os.path.join(index_dir, "communities.json")
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
            # Load inputs for standalone execution
            if target_component is not None:
                reports = utils.read_jsonl(
                    os.path.join(index_dir, "reports.jsonl")
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
            # Load inputs for standalone execution
            if target_component is not None:
                chunked_reports = utils.read_jsonl(
                    os.path.join(index_dir, "chunked_reports.jsonl")
                )

            # Build index
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
        # Input
        questions: list[Question],
        # Component-specific parameters
        top_k: int,
        # Batch API
        batch_mode: str | None = None,
        batch_dir: str | None = None,
    ) -> list[Question] | None:
        """Run all inference components and answer the questions."""

        # Retrieve contexts for each question
        contexts_list: list[ContextsForOneExample] = []
        for question in tqdm(questions, desc="Answering questions"):

            ######################################
            # [Step 6b] Passage Retrieval (Search)
            ######################################

            # Search top-k chunked reports for the question
            retrieved_chunked_reports: list[Passage] = self.passage_retrieval.search(
                queries=[question["question"]],
                top_k=top_k,
            )[0]

            # Create a ContextsForOneExample object for the question
            contexts_for_question: ContextsForOneExample = {
                "question_key": question["question_key"],
                "contexts": retrieved_chunked_reports,
            }
            contexts_list.append(contexts_for_question)

        ###############################
        # [Step 7] Answer Generation
        ###############################

        if batch_mode is None:
            results: list[Question] = []
            for question, contexts_for_question in zip(
                questions,
                contexts_list,
            ):
                # Generate the final answer
                result: Question = self.qa.answer(
                    question=question,
                    contexts_for_question=contexts_for_question,
                )

                # Preserve intermediate results
                result["contexts"] = contexts_for_question["contexts"]

                results.append(result)

        elif batch_mode == "submit":
            # Submit prompts
            batch_ids: list[str] = self.qa.submit_batch(
                questions=questions,
                contexts=contexts_list,
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
            results: list[Question] = self.qa.fetch_and_process_batch(
                questions=questions,
                contexts=contexts_list,
                batch_ids=batch_ids,
            )

            # Preserve intermediate results
            for result, contexts_for_question in zip(results, contexts_list):
                result["contexts"] = contexts_for_question["contexts"]

        else:
            raise ValueError(
                f"Invalid batch_mode: {batch_mode}. "
                "Expected None, 'submit', or 'fetch'."
            )

        if batch_mode == "submit":
            return None

        return results
