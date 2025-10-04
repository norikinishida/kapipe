from __future__ import annotations

import numpy as np
from typing import Callable
from Levenshtein import distance

from ..datatypes import Passage
from .. import utils


class TextSimilarityBasedRetriever:

    def __init__(self, normalizer: Callable[[str], str], similarity_measure: str):
        """
        A simple retriever that ranks passages based on string similarity between the query and normalized passage texts.
        Currently supports Levenshtein-based similarity.
        """
        self.normalizer = normalizer
        self.similarity_measure = similarity_measure

        if self.similarity_measure == "levenshtein":
            self.similarity_function = self.levenshtein_similarity_function
        else:
            raise Exception("Invalid `similarity_measure`: {self.similarity_measure}")

        self.passages: list[Passage] = []
        self.normalized_passages: list[str] = []
        self.caches: dict[str, np.ndarray] = {}

    def make_index(self, passages: list[Passage]) -> None:
        """
        Build index from a list of passages.
        """
        self.passages = passages
        self.normalized_passages = [
            self.normalizer(utils.create_text_from_passage(passage=p, sep=" "))
            for p in passages
        ]
        self.caches = {}

    def search(self, query: str, top_k: int = 1) -> list[Passage]:
        """
        Search for top-k similar passages based on string similarity.
        """
        scores = self.get_scores(query=query)

        # In the context of Entity Disambiguation (or Entity Linking),
        # each entity may have multiple passages (e.g., corresponding to different synonyms),
        # all sharing the same entity_id.
        # To avoid redundant matches for the same entity in the top-k results,
        # we filter out lower-ranked passages that have an entity_id already seen.
        # This ensures that the final top-k results do not contain duplicate entity_ids.
        sorted_indices = np.argsort(scores)[::-1]
        top_k_indices = []
        seen_ids = set()
        for index in sorted_indices:
            index = int(index)
            passage = self.passages[index]
            entity_id = passage.get("entity_id", index)
            if not entity_id in seen_ids:
                top_k_indices.append(index)
                seen_ids.add(entity_id)
            if len(seen_ids) >= top_k:
                break
        top_k_indices = np.asarray(top_k_indices)

        passages: list[Passage] = [self.passages[i] for i in top_k_indices]
        scores: list[float] = scores[top_k_indices]
        return [
            p | {"score": s, "rank": r+1}
            for r, (p, s) in enumerate(zip(passages, scores))
        ]

    def get_scores(self, query: str) -> np.ndarray:
        normalized_query = self.normalizer(query)

        # Cache to avoid recomputation
        if normalized_query in self.caches:
            return self.caches[normalized_query]

        scores = np.array([
            self.similarity_function(normalized_query, normalized_text)
            for normalized_text in self.normalized_passages
        ])

        self.caches[normalized_query] = scores
        return scores

    def levenshtein_similarity_function(
        self,
        normalized_query: str,
        normalized_passage: str
    ) -> float:
        """
        Compute Levenshtein-based similarity.
        """
        dist = distance(normalized_query, normalized_passage)
        return 1.0 / (dist + 1.0)
