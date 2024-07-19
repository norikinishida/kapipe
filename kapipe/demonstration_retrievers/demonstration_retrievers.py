import random

from .. import utils


class DemonstrationRetriever:
    """A demonstration (i.e., a few-shot examplers) retriever for LLM-based in-context learning methods. This retriever returns the few-shot examplers for a given document.
    """

    def __init__(
        self,
        path_demonstration_pool,
        method,
        task="docre"
    ):
        """
        Parameters
        ----------
        path_demonstration_pool : str
        method : str
        task : str, optional
            by default "docre"
        """
        assert method in ["first", "random"]

        self.path_demonstration_pool = path_demonstration_pool
        self.method = method
        self.task = task

        # Get a map from DocKey to Document
        self.demonstration_pool = {
            demo_doc["doc_key"]: demo_doc
            for demo_doc in utils.read_json(self.path_demonstration_pool)
        }

        # Get a pool of candidate document keys for retrieval
        self.doc_keys = list(self.demonstration_pool.keys())

    def retrieve(self, document, top_k, doc_keys=None):
        """
        Parameters
        ----------
        document : Document
        top_k : int
        doc_keys : list[str] | None, optional
            by default None

        Returns
        -------
        dict[str, str | list[dict[str, str | float]]]
        """
        # `doc_keys` can be used for retrieval on limited candidates
        if doc_keys is None:
            doc_keys = self.doc_keys

        # list[dict[str, str | float]]
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
        else:
            raise Exception(f"Invalid method: {self.method}")

        # dict[str, str | list[dict[str, str | float]]]
        demonstrations_for_doc = {
            "doc_key": document["doc_key"],
            "demonstrations": demonstrations_for_doc,
        }

        return demonstrations_for_doc

