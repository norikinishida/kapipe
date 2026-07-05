from __future__ import annotations

import os
import logging
from typing import Any

import networkx as nx
from tqdm import tqdm

from .. import utils
from ..chunking.base import BaseChunker
from ..community_clustering.base import BaseCommunityClusterer
from ..datatypes import CommunityRecord, ContextsForOneExample, Document, Passage, Question
from ..docre.base import BaseDocRE
from ..ed_reranking.base import BaseEDReranker
from ..ed_retrieval.base import BaseEDRetriever
from ..entity_graph_construction.base import BaseEntityGraphConstructor
from ..ner.base import BaseNER
from ..passage_retrieval.base import BasePassageRetriever
from ..qa.base import BaseQA
from ..report_generation.base import BaseReportGenerator


logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Pipeline for running GraphRAG steps separately."""

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

        self.ner = ner
        self.ed_retrieval = ed_retrieval
        self.ed_reranking = ed_reranking
        self.docre = docre
        self.entity_graph_construction = entity_graph_construction
        self.community_clustering = community_clustering
        self.report_generation = report_generation
        self.chunker = chunker
        self.passage_retrieval = passage_retrieval
        self.qa = qa

    def convert_text_to_document(
        self,
        doc_key: str,
        text: str,
        title: str | None = None,
    ) -> Document:
        """Convert raw text into a document."""

        if self.chunker is None:
            raise ValueError("Chunker is not initialized.")

        document = self.chunker.convert_text_to_document(
            doc_key=doc_key,
            text=text,
            title=title
        )

        return document

    def extract_triples(
        self,
        documents: list[Document],
        retrieval_size: int,
        index_dir: str,
    ) -> list[Document]:
        """Step 1. Extract triples from documents."""

        if self.ner is None:
            raise ValueError("NER component is not initialized.")
        if self.ed_retrieval is None:
            raise ValueError("Entity retrieval component is not initialized.")
        if self.ed_reranking is None:
            raise ValueError("Entity reranking component is not initialized.")
        if self.docre is None:
            raise ValueError("Document-level relation extraction component is not initialized.")

        # Create the output directory
        utils.mkdir(index_dir)

        # Extract triples document by document
        result_documents: list[Document] = []
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

            result_documents.append(document)

        # Save extracted triples
        utils.write_json(
            os.path.join(index_dir, "documents_with_triples.json"),
            result_documents,
        )

        logger.info(
            f"Saved extracted triples for {len(result_documents)} documents "
            f"to {index_dir}/documents_with_triples.json"
        )

        return result_documents

    def load_documents_with_triples(
        self,
        index_dir: str,
    ) -> list[Document]:
        """Load documents with triples."""

        # Load documents with extracted triples
        documents = utils.read_json(
            os.path.join(index_dir, "documents_with_triples.json"),
        )

        return documents

    def construct_entity_graph(
        self,
        documents_path_list: list[str] | None,
        entity_dict_path: str | None,
        additional_triples_path: str | None,
        index_dir: str,
    ) -> nx.MultiDiGraph:
        """Step 2. Construct an entity graph."""

        if self.entity_graph_construction is None:
            raise ValueError("Entity graph construction component is not initialized.")

        # Create the output directory
        utils.mkdir(index_dir)

        # Construct the entity graph
        graph = self.entity_graph_construction.construct_entity_graph(
            documents_path_list=documents_path_list,
            entity_dict_path=entity_dict_path,
            additional_triples_path=additional_triples_path,
        )

        # Save the entity graph
        nx.write_graphml(
            graph,
            os.path.join(index_dir, "graph.graphml"),
        )

        logger.info(f"Saved entity graph to {index_dir}/graph.graphml")

        return graph

    def load_entity_graph(
        self,
        index_dir: str,
    ) -> nx.MultiDiGraph:
        """Load an entity graph."""

        # Load the entity graph
        graph = nx.read_graphml(
            os.path.join(index_dir, "graph.graphml"),
        )

        return graph

    def cluster_communities(
        self,
        graph: nx.MultiDiGraph,
        index_dir: str,
    ) -> list[CommunityRecord]:
        """Step 3. Cluster graph communities."""

        if self.community_clustering is None:
            raise ValueError("Community clustering component is not initialized.")

        # Create the output directory
        utils.mkdir(index_dir)

        # Cluster graph communities
        communities = self.community_clustering.cluster_communities(
            graph=graph,
        )

        # Save community records
        utils.write_json(
            os.path.join(index_dir, "communities.json"),
            communities,
        )

        logger.info(
            f"Saved {len(communities)} communities to {index_dir}/communities.json"
        )

        return communities

    def load_communities(
        self,
        index_dir: str,
    ) -> list[CommunityRecord]:
        """Load community records."""

        # Load community records
        communities = utils.read_json(
            os.path.join(index_dir, "communities.json"),
        )

        return communities

    def generate_community_reports(
        self,
        graph: nx.MultiDiGraph,
        communities: list[CommunityRecord],
        index_dir: str,
        node_attr_keys: tuple[str, ...] = ("name", "entity_type", "description"),
        edge_attr_keys: tuple[str, ...] = ("relation",),
    ) -> list[Passage]:
        """Step 4. Generate community reports."""

        if self.report_generation is None:
            raise ValueError("Report generation component is not initialized.")

        # Create the output directory
        utils.mkdir(index_dir)

        # Generate community reports
        reports = self.report_generation.generate_community_reports(
            graph=graph,
            communities=communities,
            node_attr_keys=node_attr_keys,
            edge_attr_keys=edge_attr_keys,
        )

        # Save community reports
        utils.write_json(
            os.path.join(index_dir, "community_reports.json"),
            reports,
        )

        logger.info(
            f"Saved {len(reports)} community reports to "
            f"{index_dir}/community_reports.json"
        )

        return reports

    def load_community_reports(
        self,
        index_dir: str,
    ) -> list[Passage]:
        """Load community reports."""

        # Load community reports
        reports = utils.read_json(
            os.path.join(index_dir, "community_reports.json"),
        )

        return reports

    def chunk_passages(
        self,
        passages: list[Passage],
        window_size: int,
        index_dir: str,
    ) -> list[Passage]:
        """Step 5. Chunk passages."""

        if self.chunker is None:
            raise ValueError("Chunker is not initialized.")

        # Create the output directory
        utils.mkdir(index_dir)

        # Split each passage into smaller passages
        chunked_passages: list[Passage] = []
        for passage in tqdm(passages, desc="Chunking passages"):
            chunked_passages.extend(
                self.chunker.split_passage_to_chunked_passages(
                    passage=passage,
                    window_size=window_size,
                )
            )

        # Save chunked passages
        utils.write_json(
            os.path.join(index_dir, "passage_chunks.json"),
            chunked_passages,
        )

        logger.info(
            f"Saved {len(chunked_passages)} chunked passages to "
            f"{index_dir}/passage_chunks.json"
        )

        return chunked_passages

    def load_passages(
        self,
        index_dir: str,
    ) -> list[Passage]:
        """Load passage chunks."""

        # Load passage chunks
        passage_chunks = utils.read_json(
            os.path.join(index_dir, "passage_chunks.json"),
        )

        return passage_chunks

    def make_passage_retrieval_index(
        self,
        passages: list[Passage],
        index_dir: str,
        **kwargs: Any,
    ) -> None:
        """Step 6. Build a retrieval index over passages."""

        if self.passage_retrieval is None:
            raise ValueError("Passage retrieval component is not initialized.")

        # Delegate index construction to the passage retriever
        self.passage_retrieval.make_index(
            passages=passages,
            index_dir=index_dir,
            **kwargs,
        )

    def load_passage_retrieval_index(
        self,
        index_dir: str,
    ) -> None:
        """Load a retrieval index over passages."""

        # Delegate index loading to the passage retriever
        self.passage_retrieval.load_index(index_dir=index_dir)

    def infer(
        self,
        question: Question,
        top_k: int,
    ) -> Question:
        """Step 6-7. Retrieve passages and answer a question."""

        if self.passage_retrieval is None:
            raise ValueError("Passage retrieval component is not initialized.")
        if self.qa is None:
            raise ValueError("QA component is not initialized.")

        # Retrieve relevant passages
        retrieved_passages = self.passage_retrieval.search(
            queries=[question["question"]],
            top_k=top_k,
        )[0]

        # Wrap retrieved passages in QA context format
        contexts_for_question: ContextsForOneExample = {
            "question_key": question["question_key"],
            "contexts": retrieved_passages,
        }

        # Generate an answer
        result = self.qa.answer(
            question=question,
            contexts_for_question=contexts_for_question,
        )

        # Preserve retrieved contexts
        result["contexts"] = retrieved_passages

        return result