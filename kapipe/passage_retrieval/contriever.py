from __future__ import annotations

import logging
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .. import utils
from ..datatypes import Passage
from ..utils import StopWatch
from .anns import ApproximateNearestNeighborSearch


logger = logging.getLogger(__name__)


class Contriever:
    """A class for performing dense passage retrieval using the Contriever model."""
    
    def __init__(
        self,
        model_name: str = "facebook/contriever-msmarco",
        max_passage_length: int = 512,
        pooling_method: str = "average",
        normalize: bool = False,
        metric: str = "inner-product",
        device: str = "cuda",
    ) -> None:

        self.model_name = model_name
        self.max_passage_length = max_passage_length
        self.pooling_method = pooling_method
        self.normalize = normalize
        self.metric = metric
        self.device = device

        # Load the tokenizer and pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
   
        # Switch model to evaluation mode
        self.model.eval()

        # Move model to the specified device
        self.model.to(self.device)

        # Configure the model for half-precision (FP16)
        self.model = self.model.half()

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
        batch_size: int = 1024,
    ) -> None:
        """Encode passages and construct an ANN index."""

        logger.info(f"Embedding {len(passages)} passages ...")

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
        logger.info(f"Number of batches: {n_batches}")

        for batch_i, start_i in enumerate(
            range(0, len(passages), batch_size),
            1
        ):
            # Select one passage batch
            batch = passages[start_i : start_i + batch_size]

            # Combine each passage title and text
            batch_texts = [
                utils.create_text_from_passage(passage=p, sep=" ")
                for p in batch
            ]

            # Encode the batch and store its embeddings on CPU
            batch_embeddings= (
                self.encode_texts(batch_texts)
                .to(torch.float32)
                .cpu()
            )
            passage_embeddings[start_i: start_i + len(batch)] = batch_embeddings

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
        logging.info(
            "Time: %f min." % sw.get_time("passage_embedding", minute=True)
        )

        # Build the ANN index from passage embeddings
        logger.info(
            "Building index from %d passage embeddings ...",
            len(passage_embeddings),
        )
        self.anns.make_index(passage_vectors=passage_embeddings)
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

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode texts into dense vectors."""

        with torch.no_grad():

            # Tokenize the texts and move the tensors to the selected GPU
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

            # Set padding-token embeddings to zero before pooling
            token_embeddings = token_embeddings.masked_fill(
                ~attention_mask[..., None].bool(),
                0.0
            )

            # Pool the token embeddings to get text embeddings
            if self.pooling_method == "average":
                text_embeddings = (
                    token_embeddings.sum(dim=1)
                    / attention_mask.sum(dim=1)[..., None]
                )
            elif self.pooling_method == "cls":
                text_embeddings = token_embeddings[:, 0]
            else:
                raise ValueError(
                    f"Unsupported pooling method: {self.pooling_method}"
                )

            # Normalize the text embeddings if specified
            if self.normalize:
                text_embeddings = torch.nn.functional.normalize(
                    text_embeddings,
                    dim=-1
                )

            return text_embeddings

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
            "contriever",
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
            "contriever",
            "indexes",
            index_name,
        )
        logger.info(f"Loading passages and index from {index_path}")

        # Load the passages
        self.passages = utils.read_json(
            os.path.join(index_path, "passages.json")
        )

        # Load the saved ANN index if it exists;
        # otherwise, build it from passage embeddings
        index_file = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file):
            self.anns.load(index_file)
        else:
            # Rebuild the ANN index from saved passage embeddings
            logger.info(f"Index not found: {index_file}")

            # Load the passage embeddings
            logger.info(f"Loading passages embeddings from {index_path}") 
            passage_embeddings = np.load(
                os.path.join(index_path, "passage_embeddings.npy")
            )
            logger.info(f"Loaded {len(passage_embeddings)} passage embeddings")

            # Build the ANN index
            logger.info(
                "Building index from %d passage embeddings ...",
                len(passage_embeddings),
            )
            self.anns.make_index(passage_vectors=passage_embeddings)
            logger.info("Completed indexing")

            # Save the ANN index
            logger.info(f"Saving index to {index_path}")
            self.anns.save(index_file)
            logger.info("Completed saving")

        # Verify that passages and index vectors remain aligned
        if self.anns.anns_index is None:
            raise RuntimeError("Failed to load or build the ANN index")

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
                "Passages are not loaded. Call make_index() or load_index() first"
            )

        if self.anns.anns_index is None:
            raise RuntimeError(
                "ANN index is not loaded. Call make_index() or load_index() first"
            )

        # Encode queries and convert them to the ANN input format
        query_embeddings = (
            self.encode_texts(queries)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Search the ANN index for each query
        batch_indices, _, batch_scores = self.anns.search(
            query_vectors=query_embeddings,
            top_k=top_k
        )

        # Construct ranked passage objects for each query
        batch_passages: list[list[Passage]] = []
        for indices, scores in zip(batch_indices, batch_scores):
            passages_for_query: list[Passage] = []

            for i, score in zip(indices, scores):
                # Ignore the sentinel index returned when top_k exceeds index size
                if i < 0:
                    continue

                # Add the retrieval score and one-based rank
                passage = self.passages[i] | {
                    "score": float(score),
                    "rank": len(passages_for_query) + 1,
                }
                passages_for_query.append(passage)

            batch_passages.append(passages_for_query)

        return batch_passages
