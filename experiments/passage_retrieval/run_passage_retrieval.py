import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

from kapipe import utils
from kapipe.passage_retrieval import BM25, Contriever, Qwen3Embedding
from kapipe.utils import StopWatch


def main(args):
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
    input_file_path = args.input_file

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        assert args.actiontype != "search", "Prefix must be specified for search action."
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    assert actiontype in ["indexing", "search"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "passage_retrieval",
        method_name,
        config_name,
        prefix,
    )
    utils.mkdir(base_output_path)
 
    # Index will be saved to `index_dir``
    index_dir = os.path.join(base_output_path, "indexes")
    utils.mkdir(index_dir)

    # Search results will be saved to `search_results_dir`
    search_results_dir = os.path.join(base_output_path, "search_results")
    utils.mkdir(search_results_dir)

    if actiontype == "indexing":
        # Set logger
        set_logger(
            os.path.join(index_dir, "indexing.log"),
            # overwrite=True
        )

    elif actiontype == "search":
        # Set logger
        set_logger(
            os.path.join(
                search_results_dir,
                os.path.splitext(os.path.basename(input_file_path))[0] + ".log"
            ),
            # overwrite=True
        )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))
    logging.info(f"index dir: {index_dir}")

    ##################
    # Data
    ##################

    if actiontype == "indexing":
        # Load passages
        logging.info("Loading passages for indexing ...")
        passages = []
        for line in open(input_file_path):
            passage = json.loads(line.strip())
            passages.append(passage)
        logging.info(f"Loaded {len(passages)} passages")

    if actiontype == "search":
        # Load questions
        logging.info("Loading questions for search ...")
        questions = utils.read_json(input_file_path)
        logging.info(f"Loaded {len(questions)} questions")

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

    if actiontype == "indexing":
        logging.info(f"Applying the Passage Retrieval component (indexing) to passages in {input_file_path} ...")

        # Build index
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

    if actiontype == "search":
        logging.info(f"Applying the Passage Retrieval component (search) to questions in {input_file_path} ...")

        # Load the index
        retriever.load_index(index_dir=index_dir)

        # Search top-k passages for each question
        contexts = []
        batch_size = config["search_batch_size"]
        for i in tqdm(range(0, len(questions), batch_size)):
            # Create a batch of questions
            batch = questions[i:i+batch_size]

            # Search top-k passages for this batch
            batch_passages = retriever.search(
                queries=[q["question"] for q in batch],
                top_k=config["top_k"]
            )

            # Create a ContextsForOneExample object for each question
            for question, passages in zip(batch, batch_passages):
                contexts_for_question = {
                    "question_key": question["question_key"],
                    "contexts": passages
                }
                contexts.append(contexts_for_question)
 
        search_output = os.path.join(
            search_results_dir,
            os.path.splitext(os.path.basename(input_file_path))[0] + ".contexts.json"
        )
        utils.write_json(search_output, contexts)
        logging.info(f"Saved the Passage Retrieval results to {search_output}")

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))


def set_logger(filename: str, overwrite: bool =False) -> None:
    if os.path.exists(filename) and not overwrite:
        logging.info("%s already exists." % filename)
        do_remove = input("Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            logging.info("Done.")
            sys.exit(0)

    root_logger = logging.getLogger()
    handler = logging.FileHandler(filename, "w")
    root_logger.addHandler(handler)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--input_file", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Action
    parser.add_argument("--actiontype", type=str, required=True)

    args = parser.parse_args()

    main(args)