from __future__ import annotations

import logging
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .. import utils
from ..datatypes import Passage
from ..utils import StopWatch
from .anns import ApproximateNearestNeighborSearch


logger = logging.getLogger(__name__)


class Qwen3Embedding:
    """A class for performing dense passage retrieval using the Qwen3-Embedding model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_passage_length: int = 8192,
        normalize: bool = True,
        metric: str = "inner-product",
        query_instruction: str = (
            "Given a question, retrieve relevant passages "
            "that answer the question."
        ),
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.max_passage_length = max_passage_length
        self.normalize = normalize
        self.metric = metric
        self.query_instruction = query_instruction
        self.device = device

        # Load the tokenizer with left padding for last-token pooling
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )

        # Load the model with FlashAttention 2 on CUDA
        if self.device.startswith("cuda"):
            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation="flash_attention_2",
                dtype=torch.float16,
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name
            ).to(self.device)

        # Switch model to evaluation mode
        self.model.eval()

        # Initialize the FAISS-based nearest-neighbor search tool
        self.anns = ApproximateNearestNeighborSearch(
            gpu_id=-1,
            metric=self.metric,
        )

        # Cache passages after creating or loading an index
        self.passages: list[Passage] | None = None

    def make_index(
        self,
        passages: list[Passage],
        index_root: str,
        index_name: str,
        batch_size: int = 8,
    ) -> None:
        """Encode passages and construct an ANN index."""

        logger.info("Embedding %d passages ...", len(passages))

        # Start measuring passage-encoding time
        sw = StopWatch()
        sw.start("passage_embedding")

        # Obtain the embedding dimension from the loaded model
        embedding_dim: int = self.model.config.hidden_size

        # Allocate the complete embedding matrix on CPU
        passage_embeddings = torch.zeros(
            (len(passages), embedding_dim),
            dtype=torch.float32,
        )

        # Calculate the number of batches required
        n_batches = (len(passages) + batch_size - 1) // batch_size
        logger.info("Number of batches: %d", n_batches)

        for batch_i, start_i in enumerate(
            range(0, len(passages), batch_size),
            1,
        ):
            # Select one passage batch
            batch = passages[start_i : start_i + batch_size]

            # Combine each passage title and text
            batch_texts = [
                utils.create_text_from_passage(
                    passage=passage,
                    sep=" ",
                )
                for passage in batch
            ]

            # Encode passages without a query instruction
            batch_embeddings = (
                self.encode_documents(batch_texts)
                .to(torch.float32)
                .cpu()
            )
            passage_embeddings[
                start_i : start_i + len(batch)
            ] = batch_embeddings

            # Report progress periodically and after the final batch
            if batch_i % 1000 == 0 or batch_i == n_batches:
                logger.info(
                    "Processed %d/%d (%.2f%%) batches",
                    batch_i,
                    n_batches,
                    100.0 * batch_i / n_batches,
                )

        # Convert the passage embeddings to a NumPy array for FAISS
        passage_embeddings = passage_embeddings.numpy()

        # Finish measuring passage-encoding time
        logger.info("Completed passage embedding")
        sw.stop("passage_embedding")
        logger.info(
            "Time: %f min.",
            sw.get_time("passage_embedding", minute=True),
        )

        # Build the ANN index from passage embeddings
        logger.info(
            "Building index from %d passage embeddings ...",
            len(passage_embeddings),
        )
        self.anns.make_index(
            passage_vectors=passage_embeddings,
        )
        logger.info("Completed indexing")

        # Save passages, passage embeddings, and the ANN index
        self.save_index(
            passages=passages,
            passage_embeddings=passage_embeddings,
            index_root=index_root,
            index_name=index_name,
        )

        # Cache passages for retrieval without reloading the index
        self.passages = passages

    def encode_documents(
        self,
        documents: list[str],
    ) -> torch.Tensor:
        """Encode documents into dense vectors."""
        return self.encode_texts(documents)

    def encode_queries(
        self,
        queries: list[str],
    ) -> torch.Tensor:
        """Encode queries with a retrieval instruction."""

        # Add the retrieval instruction to each query
        instructed_queries = [
            f"Instruct: {self.query_instruction}\nQuery: {query}"
            for query in queries
        ]

        return self.encode_texts(instructed_queries)

    def encode_texts(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        """Encode texts into dense vectors."""

        with torch.no_grad():
            # Tokenize the texts using left padding
            model_input = self.tokenizer(
                texts,
                max_length=self.max_passage_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            # Encode the tokenized texts
            model_output = self.model(**model_input)
            token_embeddings = model_output["last_hidden_state"]
            attention_mask = model_input["attention_mask"]

            # Pool the final non-padding token
            text_embeddings = self._last_token_pool(
                last_hidden_states=token_embeddings,
                attention_mask=attention_mask,
            )

            # Normalize the text embeddings if specified
            if self.normalize:
                text_embeddings = torch.nn.functional.normalize(
                    text_embeddings,
                    p=2,
                    dim=1,
                )

            return text_embeddings

    def _last_token_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the final non-padding token."""

        # Detect whether every sequence has a valid final token
        left_padding = (
            attention_mask[:, -1].sum()
            == attention_mask.shape[0]
        )

        if left_padding:
            # Select the final token directly for left-padded inputs
            return last_hidden_states[:, -1]

        # Calculate the final non-padding position for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]

        # Select the final non-padding token for each sequence
        return last_hidden_states[
            torch.arange(
                batch_size,
                device=last_hidden_states.device,
            ),
            sequence_lengths,
        ]

    def save_index(
        self,
        passages: list[Passage],
        passage_embeddings: np.ndarray,
        index_root: str,
        index_name: str,
    ) -> None:
        """Save passages, embeddings, and an ANN index."""

        # Construct the index path and create necessary directories
        index_path = os.path.join(
            index_root,
            "qwen3_embedding",
            "indexes",
            index_name,
        )
        utils.mkdir(index_path)

        # Save passages, passage embeddings, and the ANN index
        logger.info(
            "Saving %d passages, passage embeddings, and index to %s",
            len(passages),
            index_path,
        )
        utils.write_json(
            os.path.join(index_path, "passages.json"),
            passages,
        )
        np.save(
            os.path.join(index_path, "passage_embeddings.npy"),
            passage_embeddings,
        )
        self.anns.save(
            os.path.join(index_path, "index.faiss"),
        )

        logger.info("Completed saving")

    def load_index(
        self,
        index_root: str,
        index_name: str,
    ) -> None:
        """Load an ANN index and associated passages."""

        # Construct the index directory path
        index_path = os.path.join(
            index_root,
            "qwen3_embedding",
            "indexes",
            index_name,
        )
        logger.info("Loading passages and index from %s", index_path)

        # Load the passages
        self.passages = utils.read_json(
            os.path.join(index_path, "passages.json")
        )

        # Load the saved ANN index if it exists
        index_file = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file):
            self.anns.load(index_file)
        else:
            # Report the missing ANN index
            logger.info("Index not found: %s", index_file)

            # Load the passage embeddings
            logger.info(
                "Loading passage embeddings from %s",
                index_path,
            )
            passage_embeddings = np.load(
                os.path.join(index_path, "passage_embeddings.npy")
            )
            logger.info(
                "Loaded %d passage embeddings",
                len(passage_embeddings),
            )

            # Rebuild the ANN index
            logger.info(
                "Building index from %d passage embeddings ...",
                len(passage_embeddings),
            )
            self.anns.make_index(
                passage_vectors=passage_embeddings,
            )
            logger.info("Completed indexing")

            # Save the rebuilt ANN index
            logger.info("Saving index to %s", index_file)
            self.anns.save(index_file)
            logger.info("Completed saving")

        # Verify that the ANN index was loaded or built
        if self.anns.anns_index is None:
            raise RuntimeError(
                "Failed to load or build the ANN index"
            )

        # Verify that passages and index vectors remain aligned
        if self.anns.anns_index.ntotal != len(self.passages):
            raise RuntimeError(
                "Index/passages mismatch: "
                f"{self.anns.anns_index.ntotal} vectors vs "
                f"{len(self.passages)} passages"
            )

        logger.info(
            "Completed loading %d passages and index",
            len(self.passages),
        )

    def search(
        self,
        queries: list[str],
        top_k: int = 1,
    ) -> list[list[Passage]]:
        """Retrieve the top-k passages for each query."""

        # Require passage data and an ANN index before retrieval
        if self.passages is None:
            raise RuntimeError(
                "Passages are not loaded. "
                "Call make_index() or load_index() first"
            )

        if self.anns.anns_index is None:
            raise RuntimeError(
                "ANN index is not loaded. "
                "Call make_index() or load_index() first"
            )

        # Encode queries with the retrieval instruction
        query_embeddings = (
            self.encode_queries(queries)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Search the ANN index for each query
        batch_indices, _, batch_scores = self.anns.search(
            query_vectors=query_embeddings,
            top_k=top_k,
        )

        # Construct ranked passage objects for each query
        batch_passages: list[list[Passage]] = []

        for indices, scores in zip(batch_indices, batch_scores):
            passages_for_query: list[Passage] = []

            for index, score in zip(indices, scores):
                # Ignore an invalid index returned by FAISS
                if index < 0:
                    continue

                # Add the retrieval score and one-based rank
                passage = self.passages[index] | {
                    "score": float(score),
                    "rank": len(passages_for_query) + 1,
                }
                passages_for_query.append(passage)

            batch_passages.append(passages_for_query)

        return batch_passages