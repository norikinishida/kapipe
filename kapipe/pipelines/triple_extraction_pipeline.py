from __future__ import annotations

from tqdm import tqdm

from ..chunking.base import BaseChunker
from ..datatypes import Document
from ..docre.base import BaseDocRE
from ..ed_reranking.base import BaseEDReranker
from ..ed_retrieval.base import BaseEDRetriever
from ..ner.base import BaseNER


class TripleExtractionPipeline:
    """Pipeline for chaining user-initialized triple extraction components."""

    def __init__(
        self,
        ner: BaseNER,
        ed_retrieval: BaseEDRetriever,
        ed_reranking: BaseEDReranker,
        docre: BaseDocRE,
        chunker: BaseChunker | None = None,
    ) -> None:
        self.chunker = chunker
        self.ner = ner
        self.ed_retrieval = ed_retrieval
        self.ed_reranking = ed_reranking
        self.docre = docre

    def convert_text_to_document(
        self,
        doc_key: str,
        text: str,
        title: str | None = None,
    ) -> Document:
        """Convert raw text into a document."""

        if self.chunker is None:
            raise ValueError(
                "chunker must be provided to call convert_text_to_document()."
            )

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
    ) -> list[Document]:
        """Extract triples from documents."""

        results: list[Document] = []

        for document in tqdm(documents, desc="Extracting triples from documents"):

            # Extract entity mentions
            document = self.ner.extract(document=document)

            # Retrieve candidate entities for each mention
            document, candidate_entities_for_doc = self.ed_retrieval.search(
                document=document,
                retrieval_size=retrieval_size
            )

            # Rerank candidate entities
            document = self.ed_reranking.rerank(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc
            )

            # Extract document-level relations
            document = self.docre.extract(document=document)

            results.append(document)

        return results

