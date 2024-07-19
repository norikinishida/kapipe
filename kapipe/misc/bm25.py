from collections import Counter

import numpy as np
import scipy.sparse as sp

from .. import utils


class BM25:
    """BM25, a ngram-based retriever.
    """

    def __init__(self, tokenizer, k1=1.5, b=0.75):
        """
        Parameters
        ----------
        tokenizer : callable
            a function that tokenizes a given text string into tokens
        k1 : float, optional
            by default 1.5
        b : float, optional
            by default 0.75
        """
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        # self.eps = 0.25

    def fit(self, documents):
        """
        Parameters
        ----------
        documents : list[dict]
            each dictionary has to contain an attribute "text", whose value is text.
        """
        self.documents = documents
        self.n_docs = len(documents)

        tokenized_documents = [
            self.tokenizer(doc["text"]) for doc in documents
        ]

        # Build a vocabulary by counting the frequency of each word
        counter = Counter(utils.flatten_lists(tokenized_documents))
        self.word_to_id = {
            word: wid for wid, word in enumerate(counter.keys())
        }

        # Compute the term-frequency matrix (vocab_size, n_docs)
        # Also, for each word, compute the number of documents with the word
        # (n_docs, vocab_size)
        # ---- numpy ----
        # term_freq_mat = np.zeros((self.n_docs, len(self.word_to_id)))
        # n_docs_vector = np.zeros(len(self.word_to_id)) # (vocab_size,)
        # for doc_i, tokens in enumerate(tokenized_documents):
        #     word_to_freq = Counter(tokens)
        #     for word, freq in word_to_freq.items():
        #         word_id = self.word_to_id[word]
        #         term_freq_mat[doc_i, word_id] = freq
        #         n_docs_vector[word_id] += 1
        # self.term_freq_mat = term_freq_mat
        # ---- scipy.sparse ----
        indptr = [0]
        j_indices = []
        values = []
        n_docs_vector = np.zeros(len(self.word_to_id)) # (vocab_size,)
        for doc_i, tokens in enumerate(tokenized_documents):
            word_to_freq = Counter(tokens)
            for word, freq in word_to_freq.items():
                word_id = self.word_to_id[word]
                j_indices.append(word_id)
                values.append(freq)
                n_docs_vector[word_id] += 1
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
            np.log(float(self.n_docs) - n_docs_vector + 0.5)
            - np.log(n_docs_vector + 0.5)
        ) # (vocab_size,)
        idf_vector[idf_vector < 0] = 0.0
        self.idf_vector = idf_vector

        # Compute the (average) document length
        doc_len_vector = np.zeros(self.n_docs) # (n_docs,)
        for doc_i, tokens in enumerate(tokenized_documents):
            doc_len_vector[doc_i] = len(tokens)
        avg_doc_len = np.mean(doc_len_vector)
        inv_avg_doc_len = 1.0 / avg_doc_len
        self.doc_len_vector = doc_len_vector
        self.inv_avg_doc_len = inv_avg_doc_len

        # Finally, compute the document-word weight matrix
        factor1 = self.k1 + 1.0
        factor2 = self.k1 * (
            1.0 - self.b + self.b * doc_len_vector * inv_avg_doc_len
        ) # (n_docs,)
        self.factor1 = factor1
        self.factor2 = factor2

    def search(self, query, top_k=1):
        """
        Parameters
        ----------
        query : str
        top_k : int, optional
            by default 1

        Returns
        -------
        tuple[list[str], list[str], list[float]]
        """
        scores = self.get_scores(query) # (n_docs,)

        # Since there can be documents with the same ID in the list,
        #   we need to filter out documents that have the same ID
        #   with the higher-ranked documents.
        sorted_indices = np.argsort(scores)[::-1]
        top_k_indices = []
        memory = set()
        for index in sorted_indices:
            index = int(index)
            doc = self.documents[index]
            if not doc["id"] in memory:
                top_k_indices.append(index)
                memory.add(doc["id"])
            if len(memory) >= top_k:
                break
        top_k_indices = np.asarray(top_k_indices)

        return (
            [self.documents[i]["id"] for i in top_k_indices],
            [self.documents[i]["title"] for i in top_k_indices],
            scores[top_k_indices]
        )

    def get_scores(self, query):
        """
        Parameters
        ----------
        query : str

        Returns
        -------
        list[float]
        """
        query_tokens = self.tokenizer(query)
        query_token_ids = np.asarray([self.word_to_id.get(q, -1) for q in query_tokens])
        query_token_ids = query_token_ids[query_token_ids >= 0]
        if len(query_token_ids) == 0:
            # print("Random sampling")
            return np.random.random((self.n_docs,))
        subset_idf_vector = self.idf_vector[query_token_ids] # (query_len,)
        subset_term_freq_mat = self.term_freq_mat[:, query_token_ids] # (n_docs, query_len)
        subset_term_freq_mat = subset_term_freq_mat.toarray()
        A = subset_term_freq_mat * self.factor1 # (n_docs, query_len)
        B = subset_term_freq_mat + self.factor2[:, None] # (n_docs, query_len)
        C = A / B # (n_docs, query_len)
        scores = np.dot(C, subset_idf_vector) # (n_docs,)
        return scores

