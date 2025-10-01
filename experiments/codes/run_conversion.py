import argparse
import json
import logging
import os

import sys
sys.path.insert(0, "../..")
from kapipe.chunking import Chunker
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
    spacy_model_name = args.spacy_model_name
    group_size = args.group_size

    # Input Data
    path_input_passages = args.input_passages

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
        "conversion",
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
        base_output_path + "/conversion.log",
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Method
    ##################

    # Initialize the chunker
    chunker = Chunker(model_name=spacy_model_name)

    ##################
    # Conversion from Passage to Document
    ##################

    logging.info(f"Applying the Chunking module to passages in {path_input_passages} ...")

    documents = []
    prev_number = 0
    with open(path_input_passages) as f:
        for line in f:
            # Load Passage
            passage = json.loads(line.strip())
            # Convert Passage to Document
            document = chunker.convert_passage_to_document(
                doc_key=f"Passage#{prev_number+len(documents)}",
                passage=passage,
                do_tokenize=True
            )
            documents.append(document)
            # Save the documents
            if len(documents) >= group_size:
                path_output_documents = os.path.join(
                    base_output_path,
                    f"documents.{prev_number}_{prev_number+len(documents)-1}.json"
                )
                utils.write_json(path_output_documents, documents)
                logging.info(f"Saved {prev_number}-{prev_number+len(documents)-1} documents to {path_output_documents}")
                documents = []
                prev_number += len(documents)

    if len(documents) > 0:
        path_output_documents = os.path.join(
            base_output_path,
            f"documents.{prev_number}_{prev_number+len(documents)-1}.json"
        )
        utils.write_json(path_output_documents, documents)

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
    parser.add_argument("--spacy_model_name", type=str, default="en_core_web_sm")
    parser.add_argument("--group_size", type=int, default=10000)

    # Input Data
    parser.add_argument("--input_passages", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args)
