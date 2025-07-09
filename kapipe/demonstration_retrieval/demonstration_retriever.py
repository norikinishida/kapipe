import random

from .. import utils
from ..datatypes import (
    Document,
    DemonstrationsForOneExample
)


class DemonstrationRetriever:
    """
    A demonstration (i.e., a few-shot examplers) retriever for LLM-based in-context learning methods. This retriever returns the few-shot examplers for a given document.
    """

    def __init__(
        self,
        path_demonstration_pool: str,
        method: str,
        task: str,
        path_entity_dict: str | None = None
    ):
        assert method in ["first", "random", "count"]
        assert task in ["ner", "ed", "docre"]

        self.path_demonstration_pool = path_demonstration_pool
        self.method = method
        self.task = task

        if path_entity_dict is not None:
            entity_dict = utils.read_json(path_entity_dict)
            all_entity_ids = set([e["entity_id"] for e in entity_dict])

        # Get a DocKey-Document mapping
        self.demonstration_pool = {
            demo_doc["doc_key"]: demo_doc
            for demo_doc in utils.read_json(self.path_demonstration_pool)
        }

        # Get a pool of document keys
        self.doc_keys = list(self.demonstration_pool.keys())

        # Get a list of document keys ordered based on the number of annotations
        doc_key_to_count = {}
        for doc_key, doc in self.demonstration_pool.items():
            if self.task == "ner":
                count = len(doc["mentions"])
            elif self.task == "ed":
                if path_entity_dict is not None:
                    count = sum([m["entity_id"] in all_entity_ids for m in doc["mentions"]])
                else:
                    # count = len(doc["entities"])
                    count = len(doc["mentions"])
            elif self.task == "docre":
                count = len(doc["relations"])
            else:
                count = len(doc["relations"])
            doc_key_to_count[doc_key] = count
        sorted_doc_keys = list(doc_key_to_count.items())
        self.sorted_doc_keys = sorted(sorted_doc_keys, key=lambda x: -x[1])

    def search(
        self,
        document: Document,
        top_k: int,
        doc_keys: list[str] | None = None
    ) -> DemonstrationsForOneExample:
        # `doc_keys` can be used for retrieval on limited candidates
        if doc_keys is None:
            doc_keys = self.doc_keys

        # Get demonstrations for the document
        if self.method == "first":
            demonstrations_for_doc = doc_keys[:top_k]
            demonstrations_for_doc = [
                {
                    "doc_key": key,
                    "score": 1.0
                }
                for key in demonstrations_for_doc
            ]
        elif self.method == "random":
            demonstrations_for_doc = random.sample(doc_keys, top_k)
            demonstrations_for_doc = [
                {
                    "doc_key": key,
                    "score": 1.0
                }
                for key in demonstrations_for_doc
            ]
        elif self.method == "count":
            demonstrations_for_doc = [
                {
                    "doc_key": key,
                    "score": count
                }
                for key, count in self.sorted_doc_keys[:top_k]
            ]
        else:
            raise Exception(f"Invalid method: {self.method}")

        # Create a DemonstrationsForOneExample object
        demonstrations_for_doc = {
            "doc_key": document["doc_key"],
            "demonstrations": demonstrations_for_doc,
        }

        return demonstrations_for_doc

