import numpy as np
from Levenshtein import distance


class TextSimilarityBasedRetriever:

    def __init__(self, normalizer, similarity_measure):
        """
        Parameters
        ----------
        normalizer : callable
            A function that normalizes a given text string into the normalized string
        similarity_measure : str
        """
        self.normalizer = normalizer
        self.similarity_measure = similarity_measure
        if self.similarity_measure == "levenshtein":
            self.similarity_function = self.levenshtein_similarity_function
        else:
            raise Exception("Invalid `similarity_measure`: {self.similarity_measure}")

    def fit(self, documents):
        """
        Parameters
        ----------
        documents : list[dict]
            each dictionary should contain an attribute "text", whose value is text
        """
        self.documents = documents
        self.normalized_documents = [
            self.normalizer(doc["text"]) for doc in self.documents
        ]
        self.caches = {}

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
        scores = self.get_scores(query=query)

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
        normalized_query = self.normalizer(query)
        if normalized_query in self.caches:
            return self.caches[normalized_query]
        scores = []
        for normalized_text in self.normalized_documents:
            score = self.similarity_function(normalized_query, normalized_text)
            scores.append(score)
        scores = np.asarray(scores)
        self.caches[normalized_query] = scores
        return scores

    def levenshtein_similarity_function(
        self,
        normalized_query,
        normalized_document
    ):
        """
        Parameters
        ----------
        normalized_query : str
        normalized_document : str

        Returns
        -------
        float
        """
        dist = distance(normalized_query, normalized_document)
        return 1.0 / (dist + 1.0)
