# kapipe/pipelines/rag_pipeline.py

from __future__ import annotations

from typing import Any

from ..datatypes import ContextsForOneExample, Passage, Question
from ..passage_retrieval.base import BasePassageRetriever
from ..qa.base import BaseQA


class RAGPipeline:
    """Pipeline for chaining user-initialized Passage Retrieval and QA components."""

    def __init__(
        self,
        passage_retrieval: BasePassageRetriever,
        qa: BaseQA,
    ) -> None:
        self.passage_retrieval = passage_retrieval
        self.qa = qa

    ####################
    # Indexing
    ####################

    def make_index(
        self,
        passages: list[Passage],
        index_dir: str,
        **kwargs: Any,
    ) -> None:
        """Build an index over passages."""

        self.passage_retrieval.make_index(
            passages=passages,
            index_dir=index_dir,
            **kwargs,
        )

    ####################
    # Inference
    ####################

    def load_index(
        self,
        index_dir: str,
    ) -> None:
        """Load an existing index."""
        self.passage_retrieval.load_index(index_dir=index_dir)

    def infer(
        self,
        question: Question,
        top_k: int,
    ) -> Question:
        """Retrieve passages for a question and generate an answer."""

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # Retrieve passages using the natural language question
        retrieved_passages = self.passage_retrieval.search(
            queries=[question["question"]],
            top_k=top_k,
        )[0]

        # Wrap retrieved passages in the QA context format
        contexts_for_question: ContextsForOneExample = {
            "question_key": question["question_key"],
            "contexts": retrieved_passages,
        }

        # Generate an answer using the retrieved passages
        result = self.qa.answer(
            question=question,
            contexts_for_question=contexts_for_question,
        )

        # Preserve retrieved contexts in the pipeline output
        result["contexts"] = retrieved_passages

        return result

