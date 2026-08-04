import argparse
import logging
import os
import sys
from typing import Any

import transformers

from kapipe import utils
from kapipe.datatypes import Passage
from kapipe.passage_retrieval import (
    BM25,
    BasePassageRetriever,
    Contriever,
    Qwen3Embedding,
)
from kapipe.utils import StopWatch


def main(args: argparse.Namespace) -> None:
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    method_name = args.method
    config_path = args.config_path
    config_name = args.config_name

    # Input Data
    input_passages_path = args.input_passages

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "passage_retrieval",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Index will be saved to `index_dir``
    index_dir = os.path.join(base_output_path, "indexes")
    utils.mkdir(index_dir)

    # Set logger
    set_logger(
        filename=os.path.join(index_dir, "indexing.log"),
        # overwrite=True,
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))
    logging.info(f"Index directory: {index_dir}")

    ##################
    # Data
    ##################

    # Load passages
    logging.info(f"Loading passages from {input_passages_path} ...")
    passages: list[Passage] = utils.read_jsonl(input_passages_path, encoding="utf-8")
    logging.info(f"Loaded {len(passages)} passages")

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the Passage Retrieval component
    if method_name == "bm25":
        # Instantiate the BM25-based Passage Retrieval component
        retriever = BM25(
            tokenizer=lambda text: text.lower().split(),
            k1=config["k1"],
            b=config["b"],
        )
    elif method_name == "contriever":
        # Instantiate the Contriever-based Passage Retrieval component
        retriever = Contriever(
            model_name=config["model_name"],
            max_passage_length=config["max_passage_length"],
            pooling_method=config["pooling_method"],
            normalize=config["normalize"],
            metric=config["metric"],
        )
    elif method_name == "qwen3_embedding":
        # Instantiate the Qwen3-Embedding-based Passage Retrieval component
        retriever = Qwen3Embedding(
            model_name=config["model_name"],
            max_passage_length=config["max_passage_length"],
            normalize=config["normalize"],
            metric=config["metric"],
            query_instruction=config["query_instruction"],
        )
    else:
        raise ValueError(f"Invalid retrieval method name: {method_name}")


    ##################
    # Method Execution
    ##################

    logging.info(
        f"Indexing {len(passages)} passages {input_passages_path} ..."
    )

    # Build the index
    if method_name == "bm25":
        retriever.make_index(
            passages=passages,
            index_dir=index_dir,
        )
    else:
        retriever.make_index(
            passages=passages,
            index_dir=index_dir,
            batch_size=config["indexing_batch_size"],
        )

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))


def set_logger(
    filename: str,
    overwrite: bool = False,
) -> None:
    if os.path.exists(filename) and not overwrite:
        logging.info("%s already exists." % filename)
        do_remove = input("Delete the existing log file? [y/n]: ")
        if (
            not do_remove.lower().startswith("y")
            and not len(do_remove) == 0
        ):
            logging.info("Done.")
            sys.exit(0)

    root_logger = logging.getLogger()
    handler = logging.FileHandler(
        filename,
        mode="w",
        encoding="utf-8",
    )
    root_logger.addHandler(handler)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--input_passages", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args)