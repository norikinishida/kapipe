from __future__ import annotations

import logging
import os
from typing import Any

from tqdm import tqdm

from .. import utils
from ..datatypes import ContextsForOneExample, Passage, Question
from ..passage_retrieval.base import BasePassageRetriever
from ..qa.base import BaseQA


logger: logging.Logger = logging.getLogger(__name__)


class RAGPipeline:
    """Pipeline for chaining user-initialized Passage Retrieval and QA components."""

    def __init__(
        self,
        passage_retrieval: BasePassageRetriever,
        qa: BaseQA,
    ) -> None:

        self.passage_retrieval: BasePassageRetriever = passage_retrieval
        self.qa: BaseQA = qa

    ####################
    # Indexing
    ####################

    def make_index(
        self,
        # Input
        passages: list[Passage],
        # Output directory
        index_dir: str,
        # Component-specific arguments
        passage_retrieval_indexing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Build an index over passages."""

        # Use empty mappings when optional mappings are omitted
        if passage_retrieval_indexing_kwargs is None:
            passage_retrieval_indexing_kwargs = {}

        # Create the common destination for every indexing artifact
        utils.mkdir(index_dir)

        ########################################
        # [Step 1a] Passage Retrieval (Indexing)
        ########################################

        # Build the passage retrieval index
        passage_retrieval_index_dir: str = os.path.join(
            index_dir,
            "passage_retrieval_index",
        )
        self.passage_retrieval.make_index(
            passages=passages,
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
        """Load an existing index."""
        self.passage_retrieval.load_index(
            index_dir=os.path.join(index_dir, "passage_retrieval_index"),
        )

    def infer(
        self,
        # Input
        questions: list[Question],
        # Component-specific arguments
        top_k: int,
        # Batch API
        batch_mode: str | None = None,
        batch_dir: str | None = None,
    ) -> list[Question] | None:
        """Retrieve passages for each question and generate answers."""

        # Retrieve contexts for each question
        contexts_list: list[ContextsForOneExample] = []
        for question in tqdm(questions, desc="Answering questions"):

            ######################################
            # [Step 1b] Passage Retrieval (Search)
            ######################################

            # Search top-k passages for the question
            retrieved_passages: list[Passage] = self.passage_retrieval.search(
                queries=[question["question"]],
                top_k=top_k,
            )[0]

            # Create a ContextsForOneExample object for the question
            contexts_for_question: ContextsForOneExample = {
                "question_key": question["question_key"],
                "contexts": retrieved_passages,
            }
            contexts_list.append(contexts_for_question)

        ###############################
        # [Step 2] Answer Generation
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
