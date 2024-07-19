import faiss


class ApproximateNearestNeighborSearch:
    """A wrapper for Approximate Nearest Neighbor Search library"""

    def __init__(self):
        pass

    def make_index(
        self,
        document_vectors,
        document_ids,
        document_titles
    ):
        """
        Parameters
        ----------
        document_vectors : numpy.ndarray
        document_ids : list[str] | dict[int, str]
        document_titles : list[str] | dict[int, str]
        """
        dim = document_vectors.shape[1]
        self.anns_index = faiss.IndexFlatL2(dim)
        self.anns_index.add(document_vectors)
        self.document_ids = document_ids
        self.document_titles = document_titles

    def search(self, query_vectors, retrieval_size):
        """
        Parameters
        ----------
        query_vectors : numpy.ndarray
        retrieval_size : int

        Returns
        -------
        tuple[list[list[str]], list[list[str]], list[list[float]]]
        """
        # (query_size, retrieval_size), (query_size, retrieval_size)
        batch_distances, batch_indices = self.anns_index.search(
            query_vectors,
            retrieval_size
        )
        batch_scores = 1.0 / (batch_distances + 1.0)
        batch_ids = [
            [self.document_ids[index] for index in indices]
            for indices in batch_indices
        ]
        batch_titles = [
            [self.document_titles[index] for index in indices]
            for indices in batch_indices
        ]
        return batch_ids, batch_titles, batch_scores
