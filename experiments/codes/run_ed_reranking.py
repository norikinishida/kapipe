import argparse
import logging
import os

# import numpy as np
import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.ed_reranking import EDReranking
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

    # Input Data
    path_input_documents = args.input_documents
    path_input_candidate_entities = args.input_candidate_entities

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
        "ed_reranking",
        "ed_reranking",
        identifier,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
        os.path.join(base_output_path, f"reranking.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load documents
    documents = utils.read_json(path_input_documents)

    # Load candidate entities
    candidate_entities = utils.read_json(path_input_candidate_entities)

    ##################
    # Method
    ##################

    # Initialize the ED-Reranking reranker
    reranker = EDReranking(identifier=identifier, gpu=gpu)

    ##################
    # ED-Reranking
    ##################
    
    logging.info(f"Applying the ED-Reranking component to {len(documents)} documents (+ candidate entities) in {path_input_documents} ({path_input_candidate_entities}) ...")

    # Create the full output path
    path_output_documents = os.path.join(base_output_path, f"documents.json")

    # Apply the ED-Reranking reranker to the documents and candidate entities
    result_documents = []
    for document, candidate_entities_for_doc in tqdm(
        zip(documents, candidate_entities),
        total=len(documents)
    ):
        result_document = reranker.rerank(
            document=document,
            candidate_entities_for_doc=candidate_entities_for_doc
        )
        result_documents.append(result_document)
        if len(result_documents) % 500 == 0:
            utils.write_json(path_output_documents.replace(".json", f".until_{len(result_documents)}.json"), result_documents)

    # Save the results
    utils.write_json(path_output_documents, result_documents)
    logging.info(f"Saved the prediction results to {path_output_documents}")

    # Save the prompt-response pairs visually in plain text
    if "ed_prompt" in result_documents[0] and "ed_generated_text" in result_documents[0]:
        path_output_text = os.path.join(base_output_path, "prompt_and_response.txt")
        with open(path_output_text, "w") as f:
            for doc in result_documents:
                doc_key = doc["doc_key"]
                prompt = doc["ed_prompt"]
                generated_text = doc["ed_generated_text"]
                f.write("-------------------------------------\n\n")
                f.write(f"DOC_KEY: {doc_key}\n\n")
                f.write("PROMPT:\n")
                f.write(prompt + "\n\n")
                f.write("GENERATED TEXT:\n")
                f.write(generated_text + "\n\n")
                f.flush()

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

    # Input Data
    parser.add_argument("--input_documents", type=str, required=True)
    parser.add_argument("--input_candidate_entities", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
