import argparse
import json
import logging
import os

from tqdm import tqdm

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
    window_size = args.window_size

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
        "chunking",
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
        os.path.join(base_output_path, "chunking.log"),
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
    # Chunking
    ##################

    with open(path_input_passages) as fin:
        n_lines = sum(1 for _ in fin)
    logging.info(f"Applying the Chunking component to {n_lines} passages in {path_input_passages} ...")

    # Create the full output path
    filename = os.path.splitext(os.path.basename(path_input_passages))[0]
    filename = f"{filename}.chunked_w{window_size}.jsonl"

    # Apply the chunker to the passages
    count_before = 0
    count_after = 0
    with open(os.path.join(base_output_path, filename), "w") as fout:
        with open(path_input_passages) as fin:
            for line in tqdm(fin, total=n_lines):
                # Load the passage
                passage = json.loads(line.strip())
                # Split the passage into chunked passages
                chunked_passages = chunker.split_passage_to_chunked_passages(
                    passage=passage,
                    window_size=window_size
                )
                # Save the chunked passages
                for chunked_passage in chunked_passages:
                    json_str = json.dumps(chunked_passage)
                    fout.write(json_str + "\n")
                count_before += 1
                count_after += len(chunked_passages)

    logging.info(f"Split {count_before} passages into {count_after} chunked passages")

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
    parser.add_argument("--window_size", type=int, default=100)

    # Input Data
    parser.add_argument("--input_passages", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()
    main(args) 
