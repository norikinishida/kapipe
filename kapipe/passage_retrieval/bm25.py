from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .. import utils
from ..datatypes import Passage


class BM25:
    """
    BM25 retriever: a sparse lexical matching model based on term frequency and inverse document frequency.
    """

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

    def make_index(self, passages: list[Passage]):
        """
        Build an index from a list of passages.
        """
        self.passages = passages
        self.n_passages = len(passages)

        # Tokenize the passages (texts)
        tokenized_passages = [
            self.tokenizer(utils.create_text_from_passage(passage=p, sep=" "))
            for p in passages
        ]

        # Build a vocabulary by counting the frequency of each word
        counter = Counter(utils.flatten_lists(tokenized_passages))
        self.word_to_id = {word: wid for wid, word in enumerate(counter.keys())}

        # Compute the term-frequency matrix (vocab_size, n_passages)
        # Also, for each word, compute the number of passages with the word
        # (n_passages, vocab_size)
        # ---- numpy ----
        # term_freq_mat = np.zeros((self.n_passages, len(self.word_to_id)))
        # n_passages_vector = np.zeros(len(self.word_to_id)) # (vocab_size,)
        # for p_i, tokens in enumerate(tokenized_passages):
        #     word_to_freq = Counter(tokens)
        #     for word, freq in word_to_freq.items():
        #         word_id = self.word_to_id[word]
        #         term_freq_mat[p_i, word_id] = freq
        #         n_passages_vector[word_id] += 1
        # self.term_freq_mat = term_freq_mat
        # ---- scipy.sparse ----
        indptr = [0]
        j_indices = []
        values = []
        n_passages_vector = np.zeros(len(self.word_to_id)) # (vocab_size,)
        for p_i, tokens in enumerate(tokenized_passages):
            word_to_freq = Counter(tokens)
            for word, freq in word_to_freq.items():
                word_id = self.word_to_id[word]
                j_indices.append(word_id)
                values.append(freq)
                n_passages_vector[word_id] += 1
            indptr.append(len(j_indices))
        indptr = np.asarray(indptr)
        j_indices = np.asarray(j_indices)
        values = np.asarray(values)
        self.term_freq_mat = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(self.word_to_id))
        )

        # Compute the IDF for each word
        idf_vector = (
            np.log(float(self.n_passages) - n_passages_vector + 0.5)
            - np.log(n_passages_vector + 0.5)
        ) # (vocab_size,)
        idf_vector[idf_vector < 0] = 0.0
        self.idf_vector = idf_vector

        # Compute the (average) passage length
        passage_len_vector = np.zeros(self.n_passages) # (n_passages,)
        for p_i, tokens in enumerate(tokenized_passages):
            passage_len_vector[p_i] = len(tokens)
        avg_passage_len = np.mean(passage_len_vector)
        inv_avg_passage_len = 1.0 / avg_passage_len
        self.passage_len_vector = passage_len_vector
        self.inv_avg_passage_len = inv_avg_passage_len

        # Finally, compute the passage-word weight matrix
        factor1 = self.k1 + 1.0
        factor2 = self.k1 * (
            1.0 - self.b + self.b * passage_len_vector * inv_avg_passage_len
        ) # (n_passages,)
        self.factor1 = factor1
        self.factor2 = factor2

    def search(self, query: str, top_k: int = 1) -> list[Passage]:
        scores = self.get_scores(query) # (n_passages,)

        # Since there can be passages with the same ID in the list,
        #   we need to filter out passages that have the same ID
        #   with the higher-ranked passages.
        sorted_indices = np.argsort(scores)[::-1]
        top_k_indices = []
        seen_ids = set()
        for index in sorted_indices:
            index = int(index)
            passage = self.passages[index]
            passage_id = passage.get("id", index)
            if not passage_id in seen_ids:
                top_k_indices.append(index)
                seen_ids.add(passage_id)
            if len(seen_ids) >= top_k:
                break
        top_k_indices = np.asarray(top_k_indices)

        # return (
        #     [self.passages[i]["id"] for i in top_k_indices],
        #     [self.passages[i]["title"] for i in top_k_indices],
        #     scores[top_k_indices]
        # )
        passages = [self.passages[i] for i in top_k_indices] # list[Passage]
        scores = scores[top_k_indices] # list[float]
        return [
            p | {"score": s, "rank": r+1}
            for r, (p, s) in enumerate(zip(passages, scores))
        ]
 
    def get_scores(self, query: str) -> np.ndarray:
        """
        Compute BM25 scores for a given query against all indexed passages.
        """
        query_tokens = self.tokenizer(query)
        query_token_ids = np.asarray([self.word_to_id.get(q, -1) for q in query_tokens])
        query_token_ids = query_token_ids[query_token_ids >= 0]

        if len(query_token_ids) == 0:
            # print("Random sampling")
            return np.random.random((self.n_passages,))

        # Extract sub-matrix and IDFs for query terms
        subset_idf_vector = self.idf_vector[query_token_ids] # (query_len,)
        subset_term_freq_mat = self.term_freq_mat[:, query_token_ids] # (n_passages, query_len)
        subset_term_freq_mat = subset_term_freq_mat.toarray()

        # BM25 formula: score = sum( idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len)) )
        A = subset_term_freq_mat * self.factor1 # (n_passages, query_len)
        B = subset_term_freq_mat + self.factor2[:, None] # (n_passages, query_len)
        C = A / B # (n_passages, query_len)
        scores = np.dot(C, subset_idf_vector) # (n_passages,)

        return scores

