import argparse
import logging
import os

import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.ed_retrieval import EDRetrieval
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    gpu = args.gpu
    identifier = args.identifier
    num_candidate_entities = args.num_candidate_entities

    # Input Data
    path_input_documents = args.input_documents

    # Output Path
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "ed_retrieval",
        "ed_retrieval",
        identifier,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
        os.path.join(base_output_path, "retrieval.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load documents
    documents = utils.read_json(path_input_documents)

    ##################
    # Method
    ##################

    # Initialize the ED-Retrieval retriever
    retriever = EDRetrieval(identifier=identifier, gpu=gpu)

    ##################
    # ED-Retrieval
    ##################

    logging.info(f"Applying the ED-Retrieval component to {len(documents)} documents in {path_input_documents} ...")

    # Create the full output path
    path_output_documents = os.path.join(base_output_path, "documents.json")
    path_output_candidates = os.path.join(base_output_path, "candidate_entities.json")

    # Apply the ED-Retrieval retriever to the documents
    result_documents = []
    candidate_entities = []
    for document in tqdm(documents):
        result_document, candidate_entities_for_doc = retriever.search(
            document=document,
            num_candidate_entities=num_candidate_entities
        )
        result_documents.append(result_document)
        candidate_entities.append(candidate_entities_for_doc)
        if len(result_documents) % 500 == 0:
            utils.write_json(path_output_documents.replace(".json", f".until_{len(result_documents)}.json"), result_documents)
            utils.write_json(path_output_candidates.replace(".json", f".until_{len(candidate_entities)}.json"), candidate_entities)

    # Save the results
    utils.write_json(path_output_documents, result_documents)
    utils.write_json(path_output_candidates, candidate_entities)
    logging.info(f"Saved the prediction results to {path_output_documents} and {path_output_candidates}")

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--identifier", type=str, required=True)
    parser.add_argument("--num_candidate_entities", type=int, default=10)

    # Input Data
    parser.add_argument("--input_documents", type=str, required=True)

    # Output Data
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
