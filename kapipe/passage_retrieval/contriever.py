from __future__ import annotations

import logging
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .anns import ApproximateNearestNeighborSearch
from .. import utils
from ..utils import StopWatch
from ..datatypes import Passage


logger = logging.getLogger(__name__)


class Contriever:
    """
    Passage encoder and retriever using the Contriever model.
    """
    
    def __init__(
        self,
        max_passage_length: int = 512,
        pooling_method: str = "average",
        normalize: bool = False,
        gpu_id: int = 0,
        metric: str = "inner-product"
    ):
        self.max_passage_length = max_passage_length
        self.pooling_method = pooling_method
        self.normalize = normalize
        self.gpu_id = gpu_id
        self.metric = metric

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        self.model = AutoModel.from_pretrained("facebook/contriever-msmarco")
        self.model.eval()
        self.model.to(f"cuda:{self.gpu_id}")
        self.model = self.model.half() # Convert to FP16 for efficiency

        # FAISS-based ANN search
        self.anns = ApproximateNearestNeighborSearch(gpu_id=-1, metric=self.metric)

        # Cached passages
        self.passages: list[dict] | None = None

    def encode(self, sentences: list[str]) -> torch.Tensor:
        """
        Encode a list of sentences into dense vectors.
        """
        with torch.no_grad():
            # Tokenize and tensorize the sentences
            model_input = self.tokenizer(
                sentences,
                max_length=self.max_passage_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(f"cuda:{self.gpu_id}")

            # Forward
            model_output = self.model(**model_input)
            token_embeddings = model_output["last_hidden_state"]
            attention_mask = model_input["attention_mask"]

            # Zero out embeddings of padding tokens
            token_embeddings = token_embeddings.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )

            # Pooling
            if self.pooling_method == "average":
                sentence_embeddings = (
                    token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                )
            elif self.pooling_method == "cls":
                sentence_embeddings = token_embeddings[:, 0]
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

            if self.normalize:
                sentence_embeddings = torch.nn.functional.normalize(
                    sentence_embeddings,
                    dim=-1
                )

            return sentence_embeddings

    def make_index(
        self,
        passages: list[Passage],
        index_root: str,
        index_name: str,
        batch_size: int = 1024
    ) -> None:
        """
        Compute embeddings and construct ANN index from passages.
        """
        logger.info(f"Embedding {len(passages)} passages ...")
        sw = StopWatch()
        sw.start("main")

        # passage_embeddings = []
        dim = 768 # default hidden size for Contriever

        # Initialize the embeddings matrix
        passage_embeddings = torch.zeros((len(passages), dim), dtype=torch.float32)

        # Calculate the number of batches required
        n_batches = (len(passages) + batch_size - 1) // batch_size
        logger.info(f"Number of batches: {n_batches}")

        for batch_i, start_i in enumerate(range(0, len(passages), batch_size), 1):
            # Get batch input
            batch = passages[start_i : start_i + batch_size]
            batch_texts = [
                utils.create_text_from_passage(passage=p, sep=" ") for p in batch
            ]
            # Encode the batch
            embs = self.encode(batch_texts).to(torch.float32).cpu()
            passage_embeddings[start_i: start_i + len(batch)] = embs
            # Show status
            if (batch_i % 1000 == 0) or (batch_i == n_batches):
                logger.info(f"Processed {batch_i}/{n_batches} ({100.0*batch_i/n_batches:.2f}%) batches")

        passage_embeddings = passage_embeddings.numpy()

        logger.info("Completed passage embedding")
        sw.stop("main")
        logging.info("Time: %f min." % sw.get_time("main", minute=True))

         # Make index
        logger.info(f"Building index from {len(passage_embeddings)} passage embeddings ...")
        self.anns.make_index(passage_vectors=passage_embeddings)
        logger.info("Completed indexing")

         # Save the passages, passage embeddings, and index
        index_path = os.path.join(index_root, "contriever", "indexes", index_name)
        utils.mkdir(index_path)
        logger.info(f"Saving {len(passage_embeddings)} passages, passage embeddings, and index to {index_path}")
        utils.write_json(os.path.join(index_path, "passages.json"), passages)
        np.save(os.path.join(index_path, "passage_embeddings.npy"),  passage_embeddings)
        self.anns.save(os.path.join(index_path, "index.faiss"))
        logger.info("Completed saving")

        self.passages = passages
        
    def load_index(self, index_root: str, index_name: str) -> None:
        """
        Load passage data and ANN index from disk.
        """
        index_path = os.path.join(index_root, "contriever", "indexes", index_name)
        logger.info(f"Loading passages and index from {index_path}")

        # Load the passages
        self.passages = utils.read_json(os.path.join(index_path, "passages.json"))

        index_file = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file):
            # Load the index
            self.anns.load(index_file)
        else:
            logger.info(f"Index not found: {index_file}")
            # Load the passage embeddings
            logger.info(f"Loading passages embeddings from {index_path}") 
            passage_embeddings = np.load(
                os.path.join(index_path, "passage_embeddings.npy")
            )
            logger.info(f"Loaded {len(passage_embeddings)} passage embeddings")
            # Make index
            logger.info(f"Building index from {len(passage_embeddings)} passage embeddings ...")
            self.anns.make_index(passage_vectors=passage_embeddings)
            logger.info("Completed indexing")
            # save the index
            logger.info(f"Saving index to {index_path}")
            self.anns.save(index_file)
            logger.info("Completed saving")

        logger.info(f"Complete loading {len(self.passages)} passages and index")

    def search(self, queries: list[str], top_k: int = 1) -> list[list[Passage]]:
        """
        Search for the most relevant passages for each query.
        """
        # Encode the queries
        query_embeddings = self.encode(queries).cpu().numpy().astype(np.float32)
        # Search top-k passages for each query
        batch_indices, _, _, batch_scores = self.anns.search(
            query_vectors=query_embeddings,
            top_k=top_k
        )
        # Create output passages with scores and ranks
        batch_passages = [
            [
                self.passages[i] | {"score": float(s), "rank": r+1}
                for r, (i,s) in enumerate(zip(indices, scores))
            ]
            for indices, scores in zip(batch_indices, batch_scores)
        ]
        return batch_passages
