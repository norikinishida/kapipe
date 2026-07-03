from __future__ import annotations

import logging
import math

import faiss
import numpy as np


logger = logging.getLogger(__name__)


class ApproximateNearestNeighborSearch:
    """A class for performing approximate nearest-neighbor search using FAISS."""

    def __init__(self, metric: str = "l2", gpu_id: int = -1) -> None:
        self.metric = metric
        self.gpu_id = gpu_id

        # Initialize the ANN index and passage metadata
        self.anns_index: faiss.Index | None = None
        self.passage_metadatas: list[dict] | None = None

    def make_index(
        self,
        passage_vectors: np.ndarray,
        passage_metadatas: list[dict] | None = None,
    ) -> None:
        """Build an ANN index from passage vectors."""

        # Obtain the vector dimension
        dim = passage_vectors.shape[1]

        # Select the FAISS index according to the distance metric
        if self.metric == "inner-product":
            # Follow the index used by the original Contriever implementation:
            # https://github.com/facebookresearch/contriever/blob/main/src/index.py
            self.anns_index = faiss.IndexFlatIP(dim)

        elif self.metric == "hnsw-inner-product":
            # Configure the HNSW graph
            store_n = 128 # neighbors to store per node
            ef_search = 128 # search depth
            # ef_construction = 200 # construction time search depth

            # Build an HNSW index using inner-product similarity
            self.anns_index = faiss.IndexHNSWFlat(
                dim, store_n, faiss.METRIC_INNER_PRODUCT
            )
            self.anns_index.hnsw.efSearch = ef_search
            # self.anns_index.hnsw.efConstruction = ef_construction

        else:
            # Use squared L2 distance for other metric names
            self.anns_index = faiss.IndexFlatL2(dim)

        # Move the index to the specified GPU if requested
        if self.gpu_id >= 0:
            logger.info(f"Converting CPU index to GPU index (GPU ID: {self.gpu_id}) ...")

            # Initialize GPU resources
            res = faiss.StandardGpuResources()

            # Configure FP16 storage on the GPU
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True

            # Copy the CPU index to the selected GPU
            self.anns_index = faiss.index_cpu_to_gpu(
                res, self.gpu_id, self.anns_index, co
            )
        else:
            logger.info("Using CPU index mode")

        # Add passage vectors in batches to limit memory usage
        INDEXING_BATCH_SIZE = 1000000
        n_iterations = math.ceil(len(passage_vectors) / INDEXING_BATCH_SIZE)
        it = 1
        for i in range(0, len(passage_vectors), INDEXING_BATCH_SIZE):
            logger.info(
                "Iteration [%d/%d]: Indexing %d-%d passage embeddings",
                it,
                n_iterations,
                i,
                min(
                    i + INDEXING_BATCH_SIZE,
                    len(passage_vectors),
                ) - 1,
            )

            # Add one batch of passage vectors
            self.anns_index.add(passage_vectors[i : i + INDEXING_BATCH_SIZE])
            it += 1

        # Cache passage metadata in the same order as the vectors
        self.passage_metadatas = passage_metadatas

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 1,
        batch_size: int = 1024,
    ) -> tuple[
        list[list[int]],
        list[list[dict]] | None,
        list[list[float]],
    ]:
        """Retrieve the top-k passages for each of the query vectors."""

        # Require a built or loaded index before searching
        if self.anns_index is None:
            raise RuntimeError(
                "ANN index is not available. "
                "Call make_index() or load() first"
            )

        # Require a positive retrieval size
        if top_k <= 0:
            raise ValueError(
                f"top_k must be positive: {top_k}"
            )

        # Require a positive query batch size
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive: {batch_size}"
            )

        # Require at least one indexed passage vector
        n_passages = self.anns_index.ntotal
        if n_passages == 0:
            raise RuntimeError(
                "ANN index contains no passage vectors"
            )

        # Prevent FAISS from returning -1 for missing neighbors
        if top_k > n_passages:
            logger.warning(
                "Reducing top_k from %d to %d because "
                "the index contains only %d passage vectors",
                top_k,
                n_passages,
                n_passages,
            )
            top_k = n_passages

        n_queries = len(query_vectors)

        # Initialize outputs for all queries
        all_indices: list[list[int]] = []
        all_metadatas: list[list[dict]] | None = (
            []
            if self.passage_metadatas is not None
            else None
        )
        all_scores: list[list[float]] = []

        # Process query vectors in batches
        for begin_i in range(0, n_queries, batch_size):
            # Select one query batch
            end_i = min(begin_i + batch_size, n_queries)
            batch_vectors = query_vectors[begin_i: end_i]

            # Search the top-k passages for each query
            # (batch_size, top_k), (batch_size, top_k)
            batch_scores, batch_indices = self.anns_index.search(batch_vectors, top_k)

            # Convert L2 distances to similarity scores
            if self.metric not in {"inner-product", "hnsw-inner-product"}:
                batch_scores = 1.0 / (batch_scores + 1.0)

            # Convert NumPy outputs to Python lists
            batch_indices = batch_indices.tolist()
            batch_scores = batch_scores.tolist()

            # Append the current batch outputs
            all_indices.extend(batch_indices)
            all_scores.extend(batch_scores)

            # Map passage indices to metadata when available
            if self.passage_metadatas is not None:
                batch_metadatas = [
                    [
                        self.passage_metadatas[i]
                        for i in indices
                    ]
                    for indices in batch_indices
                ]

                # This branch guarantees that all_metadatas is a list
                assert all_metadatas is not None
                all_metadatas.extend(batch_metadatas)

        return all_indices, all_metadatas, all_scores

    def save(self, path: str) -> None:
        """Save the ANN index."""

        # Require a built or loaded index before saving
        if self.anns_index is None:
            raise RuntimeError(
                "ANN index is not available. "
                "Call make_index() or load() first"
            )

        # Save the FAISS index
        faiss.write_index(self.anns_index, path)

    def load(self, path: str) -> None:
        """Load an ANN index."""

        # Load the FAISS index
        self.anns_index = faiss.read_index(path)