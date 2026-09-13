from __future__ import annotations

from collections import Counter
import os
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .. import utils
from ..datatypes import Passage
from .base import BasePassageRetriever


class BM25(BasePassageRetriever):
    """A class for performing sparse lexical passage retrieval using BM25."""

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]],
        k1: float = 1.5,
        b: float = 0.75
    ) -> None:

        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        # self.eps = 0.25

        # Initialize index data
        self.passages: list[Passage] | None = None
        self.n_passages: int | None = None
        self.word_to_id: dict[str, int] | None = None
        self.term_freq_mat: sp.csr_matrix | None = None
        self.idf_vector: np.ndarray | None = None
        self.passage_len_vector: np.ndarray | None = None
        self.inv_avg_passage_len: float | None = None
        self.factor1: float | None = None
        self.factor2: np.ndarray | None = None

    def make_index(
        self,
        passages: list[Passage],
        index_dir: str,
    ) -> None:
        """Build a BM25 index from passages."""

        # Store passages in their index order
        self.passages = passages
        self.n_passages = len(passages)

        # Tokenize passage titles and texts
        tokenized_passages = [
            self.tokenizer(
                utils.create_text_from_passage(passage=p, sep=" ")
            )
            for p in passages
        ]

        # Build the vocabulary
        counter = Counter(utils.flatten_lists(tokenized_passages))
        self.word_to_id = {
            word: wid
            for wid, word in enumerate(counter.keys())
        }

        # Initialize data for the sparse term-frequency matrix
        indptr: list[int] = [0]
        j_indices: list[int] = []
        values: list[int] = []

        # Count the number of passages containing each word
        # (vocab_size,)
        n_passages_vector = np.zeros(len(self.word_to_id))
        for p_i, tokens in enumerate(tokenized_passages):
            # Count word frequencies in one passage
            word_to_freq = Counter(tokens)

            # Store the nonzero term frequencies
            for word, freq in word_to_freq.items():
                word_id = self.word_to_id[word]
                j_indices.append(word_id)
                values.append(freq)
                n_passages_vector[word_id] += 1

            # Record the end of the current sparse row
            indptr.append(len(j_indices))

        # Convert sparse matrix data to NumPy arrays
        indptr = np.asarray(indptr)
        j_indices = np.asarray(j_indices)
        values = np.asarray(values)

        # construct the passage-term frequency matrix
        self.term_freq_mat = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(self.word_to_id))
        )

        # Compute the IDF for each word
        # (vocab_size,)
        idf_vector = (
            np.log(float(self.n_passages) - n_passages_vector + 0.5)
            - np.log(n_passages_vector + 0.5)
        )

        # Clip negative IDF values to zero
        idf_vector[idf_vector < 0] = 0.0
        self.idf_vector = idf_vector

        # Compute the inverse average passage length
        # (n_passages,)
        passage_len_vector = np.zeros(self.n_passages)
        for p_i, tokens in enumerate(tokenized_passages):
            passage_len_vector[p_i] = len(tokens)

        # Compute the inverse average passage length
        avg_passage_len = np.mean(passage_len_vector)
        inv_avg_passage_len = 1.0 / avg_passage_len

        self.passage_len_vector = passage_len_vector
        self.inv_avg_passage_len = inv_avg_passage_len

        # Precompute query-independent BM25 factors
        # (n_passages,)
        factor1 = self.k1 + 1.0
        factor2 = self.k1 * (
            1.0 - self.b + self.b * passage_len_vector * inv_avg_passage_len
        )
        self.factor1 = factor1
        self.factor2 = factor2

        # Save the built index
        self.save_index(index_dir=index_dir)

    def save_index(
        self,
        index_dir: str,
    ) -> None:
        """Save the built index."""

        # Validate that the BM25 index has been built before saving
        if (
            self.passages is None
            or self.word_to_id is None
            or self.term_freq_mat is None
            or self.idf_vector is None
            or self.passage_len_vector is None
            or self.inv_avg_passage_len is None
            or self.factor1 is None
            or self.factor2 is None
        ):
            raise RuntimeError(
                "BM25 index is not built. Call make_index() first"
            )

        # Create the output directory
        utils.mkdir(index_dir)

        # Save passages
        utils.write_json(
            os.path.join(index_dir, "passages.json"),
            self.passages,
        )

        # Save vocabulary
        utils.write_json(
            os.path.join(index_dir, "word_to_id.json"),
            self.word_to_id,
        )

        # Save the sparse term-frequency matrix
        sp.save_npz(
            os.path.join(index_dir, "term_freq_mat.npz"),
            self.term_freq_mat,
        )

        # Save dense arrays and scalar values
        np.savez(
            os.path.join(index_dir, "bm25.npz"),
            idf_vector=self.idf_vector,
            passage_len_vector=self.passage_len_vector,
            inv_avg_passage_len=np.asarray(self.inv_avg_passage_len),
            factor1=np.asarray(self.factor1),
            factor2=self.factor2,
        )

    def load_index(
        self,
        index_dir: str,
    ) -> None:
        """Load the BM25 index."""

        # Load passages
        self.passages = utils.read_json(
            os.path.join(index_dir, "passages.json")
        )

        # Restore the number of passages
        self.n_passages = len(self.passages)

        # Load vocabulary
        self.word_to_id = utils.read_json(
            os.path.join(index_dir, "word_to_id.json")
        )

        # Load the sparse term-frequency matrix
        self.term_freq_mat = sp.load_npz(
            os.path.join(index_dir, "term_freq_mat.npz")
        )

        # Load dense arrays and scalar values
        arrays = np.load(
            os.path.join(index_dir, "bm25.npz")
        )

        # Restore BM25 index data
        self.idf_vector = arrays["idf_vector"]
        self.passage_len_vector = arrays["passage_len_vector"]
        self.inv_avg_passage_len = float(arrays["inv_avg_passage_len"])
        self.factor1 = float(arrays["factor1"])
        self.factor2 = arrays["factor2"]

    def search(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[list[Passage]]:
        """Retrieve the top-k passages for a batch of queries."""

        batch_passages: list[list[Passage]] = []
        for query in queries:
            passages = self._search_one(
                query=query,
                top_k=top_k
            )
            batch_passages.append(passages)

        return batch_passages

    def _search_one(self, query: str, top_k: int) -> list[Passage]:
        """Retrieve the top-k passages for a single query."""

        # Validate that the BM25 index has been built before retrieval
        if self.passages is None:
            raise RuntimeError(
                "Passages are not indexed. Call make_index() first"
            )

        # Compute scores for all passages
        # (n_passages,)
        scores = self.get_scores(query)

        # In the context of Entity Disambiguation (or Entity Linking),
        # each entity may have multiple passages (e.g., corresponding to different synonyms),
        # all sharing the same entity_id.
        # To avoid redundant matches for the same entity in the top-k results,
        # we filter out lower-ranked passages that have an entity_id already seen.
        # This ensures that the final top-k results do not contain duplicate entity_id.

        # Sort passages by descending scores
        sorted_indices = np.argsort(scores)[::-1]

        # Keep only the highest-ranked passage for each unique ID
        top_k_indices = []
        seen_ids = set()
        for index in sorted_indices:
            index = int(index)
            passage = self.passages[index]
            entity_id = passage.get("entity_id", index)

            if entity_id not in seen_ids:
                top_k_indices.append(index)
                seen_ids.add(entity_id)

            if len(seen_ids) >= top_k:
                break

        # Construct ranked passage objects
        return [
            self.passages[index] | {
                "score": float(scores[index]),
                "rank": r + 1,
            }
            for r, index in enumerate(top_k_indices)
        ]
 
    def get_scores(
        self,
        query: str,
    ) -> np.ndarray:
        """Compute BM25 scores for all indexed passages."""

        # Validate that the BM25 index has been built before scoring
        if (
            self.n_passages is None
            or self.word_to_id is None
            or self.term_freq_mat is None
            or self.idf_vector is None
            or self.factor1 is None
            or self.factor2 is None
        ):
            raise RuntimeError(
                "BM25 index is not built. Call make_index() first"
            )

        # Tokenize the query
        query_tokens = self.tokenizer(query)

        # Map known query words to vocabulary IDs
        query_token_ids = np.asarray(
            [
                self.word_to_id.get(q, -1)
                for q in query_tokens
            ]
        )
        query_token_ids = query_token_ids[query_token_ids >= 0]

        # Use random scores when the query contains no known words
        if len(query_token_ids) == 0:
            return np.random.random((self.n_passages,))

        # Extract IDFs for query words
        # (query_len,)
        subset_idf_vector = self.idf_vector[query_token_ids]

        # Extract term frequencies for query words
        # (n_passages, query_len)
        subset_term_freq_mat = self.term_freq_mat[:, query_token_ids]
        subset_term_freq_mat = subset_term_freq_mat.toarray()

        # Compute BM25 scores
        # score = sum( idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len)) )
        # (n_passages, query_len)
        A = subset_term_freq_mat * self.factor1
        # (n_passages, query_len)
        B = subset_term_freq_mat + self.factor2[:, None]
        # (n_passages, query_len)
        C = A / B
        # (n_passages,)
        scores = np.dot(C, subset_idf_vector)

        return scores

