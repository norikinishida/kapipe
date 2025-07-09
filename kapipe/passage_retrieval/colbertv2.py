from __future__ import annotations

import logging
import os

from ragatouille import RAGPretrainedModel

from .. import utils
from ..datatypes import Passage


logger = logging.getLogger(__name__)


class ColBERTv2Retriever:
    """
    Passage retriever using ColBERTv2 via RAGatouille interface.
    """
    
    def __init__(self):
        self.rag: RAGPretrainedModel | None = None

    def make_index(
        self,
        passages: list[Passage],
        index_root: str,
        index_name: str,
        max_passage_length: int = 256,
        use_faiss: bool = True
    ) -> None:
        """
        Build ColBERTv2 index from a list of passages.
        """
        # Initialize the RAGPretrainedModel object
        self.rag = RAGPretrainedModel.from_pretrained(
            pretrained_model_name_or_path="colbert-ir/colbertv2.0",
            index_root=index_root,
            n_gpu=1
        )

        # Convert each passage (dict) to text
        passage_texts = [
            utils.create_text_from_passage(passage=p, sep=" ") for p in passages
        ]

        # Make index
        logger.info("Building index ...")
        self.rag.index(
            index_name=index_name,
            collection=passage_texts,
            document_ids=None,
            document_metadatas=passages,
            overwrite_index=False,
            max_document_length=max_passage_length,
            use_faiss=use_faiss
        )
        logger.info("Completed indexing")
        logger.info(f"Saved index to {os.path.join(index_root, 'colbert/indexes', index_name)}")

    def load_index(self, index_root: str, index_name: str) -> None:
        """
        Load an existing ColBERTv2 index from disk.
        """
        index_path = os.path.join(index_root, "colbert/indexes", index_name)
        logger.info(f"Loading index from {index_path}")
        self.rag = RAGPretrainedModel.from_index(
            index_path=index_path,
            n_gpu=1
        )
        logger.info("Completed loading")

    def search(self, queries: list[str], top_k: int = 1) -> list[list[Passage]]:
        """
        Search for the most relevant passages for each query.
        """
        batch_results = self.rag.search(query=queries, k=top_k)

        # Ensure batch_results is always a list of lists
        if len(queries) == 1:
            batch_results = [batch_results]

        # Create output passages with scores and ranks
        batch_passages: list[list[Passage]] = []
        for results in batch_results:
            ranked_passages: list[Passage] = []
            for result in results:
                # Retrieve original metadata and enrich it with ranking information
                passage = result["document_metadata"]
                passage["score"] = result["score"]
                passage["rank"] = result["rank"]
                ranked_passages.append(passage)
            batch_passages.append(ranked_passages)
        return batch_passages


            
