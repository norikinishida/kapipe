from __future__ import annotations

import logging
import os

from tqdm import tqdm

from .. import utils
from ..chunking.base import BaseChunker
from ..datatypes import CandidateEntitiesForDocument, Document
from ..docre.base import BaseDocRE
from ..ed_reranking.base import BaseEDReranker
from ..ed_retrieval.base import BaseEDRetriever
from ..ner.base import BaseNER


logger: logging.Logger = logging.getLogger(__name__)


class TripleExtractionPipeline:
    """Pipeline for chaining user-initialized triple extraction components."""

    def __init__(
        self,
        ner: BaseNER,
        ed_retrieval: BaseEDRetriever,
        ed_reranking: BaseEDReranker,
        docre: BaseDocRE,
        chunker: BaseChunker,
    ) -> None:

        self.ner: BaseNER = ner
        self.ed_retrieval: BaseEDRetriever = ed_retrieval
        self.ed_reranking: BaseEDReranker = ed_reranking
        self.docre: BaseDocRE = docre
        self.chunker: BaseChunker = chunker

    def convert_text_to_document(
        self,
        doc_key: str,
        text: str,
        title: str | None = None,
    ) -> Document:
        """Convert raw text into a document."""

        # Convert raw text into a document
        document: Document = self.chunker.convert_text_to_document(
            doc_key=doc_key,
            text=text,
            title=title,
        )

        return document

    def extract_triples(
        self,
        # Input
        documents: list[Document],
        # Component-specific arguments
        retrieval_size: int,
        # Target component
        target_component: str | None = None,
        intermediate_dir: str | None = None,
        # Batch API
        batch_mode: str | None = None,
        batch_dir: str | None = None,
    ) -> list[Document] | None:
        """Extract triples from documents or run one selected component."""

        # Validate the target component
        valid_target_components: list[str] = [
            "ner",
            "ed_retrieval",
            "ed_reranking",
            "docre",
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

        # Create the common destination for every intermediate artifact
        if target_component is not None:
            utils.mkdir(intermediate_dir)

        ##############################
        # [Step 1] NER
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
                if target_component is not None:
                    # Save documents with extracted mentions as intermediate artifacts
                    utils.write_json(
                        os.path.join(intermediate_dir, "documents_with_mentions.json"),
                        documents_with_mentions,
                    )

        ##############################
        # [Step 2] ED-Retrieval
        ##############################

        # Retrieve candidate entities for the extracted mentions
        if target_component is None or target_component == "ed_retrieval":
            # Load inputs for standalone execution
            if target_component is not None:
                documents_with_mentions: list[Document] = utils.read_json(
                    os.path.join(intermediate_dir, "documents_with_mentions.json")
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

            if target_component == "ed_retrieval":
                # Save documents and candidate entities for reranking 
                # as intermediate artifacts.
                utils.write_json(
                    os.path.join(intermediate_dir, "documents_with_candidates.json"),
                    documents_with_candidates,
                )
                utils.write_json(
                    os.path.join(intermediate_dir, "candidate_entities.json"),
                    candidate_entities,
                )

        ##############################
        # [Step 3] ED-Reranking
        ##############################

        # Rerank candidate entities for the extracted mentions
        if target_component is None or target_component == "ed_reranking":
            # Load inputs for standalone execution
            if target_component is not None:
                documents_with_candidates: list[Document] = utils.read_json(
                    os.path.join(intermediate_dir, "documents_with_candidates.json")
                )
                candidate_entities: list[CandidateEntitiesForDocument] = (
                    utils.read_json(
                        os.path.join(intermediate_dir, "candidate_entities.json")
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
                if target_component is not None:
                    # Save documents with disambiguated entities 
                    # as intermediate artifacts.
                    utils.write_json(
                        os.path.join(intermediate_dir, "documents_with_entities.json"),
                        documents_with_entities,
                    )

        ##############################
        # [Step 4] DocRE
        ##############################

        # Extract document-level relations
        if target_component is None or target_component == "docre":
            # Load inputs for standalone execution
            if target_component is not None:
                documents_with_entities: list[Document] = utils.read_json(
                    os.path.join(intermediate_dir, "documents_with_entities.json")
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

            if batch_mode == "submit":
                return None

            return documents_with_triples
