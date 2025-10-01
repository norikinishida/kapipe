import argparse
import json
import logging
import os

from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.passage_retrieval import Contriever
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    gpu_id = args.gpu
    method_name = args.method
    metric_name = args.metric
    top_k = args.top_k

    # Input Data
    path_input_file = args.input_file

    # Output Path
    path_results_dir = args.results_dir
    index_name = args.index_name
    if index_name is None or index_name == "None":
        index_name = utils.get_current_time()
        args.index_name = index_name

    # Action
    actiontype = args.actiontype

    assert actiontype in ["indexing", "search"]

    ##################
    # Logging Setup
    ##################

    # Set index root
    index_root = os.path.join(
        path_results_dir,
        "passage_retrieval"
    )
    utils.mkdir(os.path.join(
        index_root,
        method_name
    ))
  
    if actiontype == "indexing":
        # Set base output path
        utils.mkdir(os.path.join(
            index_root,
            method_name,
            "indexes",
            index_name
        ))
        # Set logger
        shared_functions.set_logger(
            os.path.join(
                index_root,
                method_name,
                "indexes",
                index_name,
                "indexing.log"  
            ),
            # overwrite=True
        )
    elif actiontype == "search":
        # Set base output path
        utils.mkdir(os.path.join(
            index_root,
            method_name,
            "search-results",
            index_name
        ))
        # Set logger
        shared_functions.set_logger(
            os.path.join(
                index_root,
                method_name,
                "search-results",
                index_name,
                os.path.splitext(os.path.basename(path_input_file))[0] + ".log"
            ),
            # overwrite=True
        )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    # Index will be saved to index_root/method_name/indexes/index_name/...
    # Search results will be saved to index_root/method_name/search-results/index_name/...
    # Logs will be saved to index_root/method_name/{indexing,search}.log

    logging.info(f"index_root: {index_root}")
    logging.info(f"index_name: {index_name}")

    ##################
    # Data
    ##################

    if actiontype == "indexing":
        # Load passages
        logging.info("Loading passages for indexing ...")
        passages = []
        for line in open(path_input_file):
            passage = json.loads(line.strip())
            passages.append(passage)
        logging.info(f"Loaded {len(passages)} passages")

    if actiontype == "search":
        # Load questions
        logging.info("Loading questions for search ...")
        questions = utils.read_json(path_input_file)
        logging.info(f"Loaded {len(questions)} questions")

    ##################
    # Method
    ##################

    # Initialize the passage retriever
    if method_name == "contriever":
        retriever = Contriever(
            max_passage_length=512,
            pooling_method="average",
            normalize=False,
            gpu_id=gpu_id,
            metric=metric_name
        )
    else:
        raise ValueError(f"Invalid retrieval method name: {method_name}")

    ##################
    # Indexing, Search
    ##################


    if actiontype == "indexing":
        logging.info(f"Applying the Passage Retrieval module (indexing) to passages in {path_input_file} ...")

        # Build index
        retriever.make_index(
            passages=passages,
            index_root=index_root,
            index_name=index_name
        )

    if actiontype == "search":
        logging.info(f"Applying the Passage Retrieval module (search) to questions in {path_input_file} ...")

        # Load the index
        retriever.load_index(index_root=index_root, index_name=index_name)

        # Search top-k passages for each question
        contexts = []
        BATCH_SIZE = 10
        for i in tqdm(range(0, len(questions), BATCH_SIZE)):
            # Create a batch of questions
            batch = questions[i:i+BATCH_SIZE]
            # Search top-k passages for this batch
            batch_passages = retriever.search(
                queries=[q["question"] for q in batch],
                top_k=top_k
            )
            # Create a ContextsForOneExample object for each question
            for question, passages in zip(batch, batch_passages):
                contexts_for_question = {
                    "question_key": question["question_key"],
                    "contexts": passages
                }
                contexts.append(contexts_for_question)
 
        search_output = os.path.join(
            index_root,
            method_name,
            "search-results",
            index_name,
            os.path.splitext(os.path.basename(path_input_file))[0] + ".contexts.json"
        )
        utils.write_json(search_output, contexts)
        logging.info(f"Saved the retrieval results to {search_output}")

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)

    # Input Data
    parser.add_argument("--input_file", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--index_name", type=str, default=None)

    # Action
    parser.add_argument("--actiontype", type=str, required=True)

    args = parser.parse_args()

    main(args)