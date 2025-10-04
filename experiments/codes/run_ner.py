import argparse
import logging
import os

import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.ner import NER
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
        "ner",
        "ner",
        identifier,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
        os.path.join(base_output_path, "extraction.log"),
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

    # Initialize the NER extractor
    extractor = NER(identifier=identifier, gpu=gpu)

    ##################
    # NER
    ##################

    logging.info(f"Applying the NER component to {len(documents)} documents in {path_input_documents} ...")

    # Create the full output path
    path_output_documents = os.path.join(base_output_path, "documents.json")

    # Apply the NER extractor to the documents
    result_documents = []
    for document in tqdm(documents):
        result_document = extractor.extract(document=document)
        result_documents.append(result_document)
        if len(result_documents) % 500 == 0:
            utils.write_json(path_output_documents.replace(".json", f".until_{len(result_documents)}.json"), result_documents)

    # Save the results
    utils.write_json(path_output_documents, result_documents)
    logging.info(f"Saved the prediction results to {path_output_documents}")

    # Save the prompt-response pairs visually in plain text
    if "ner_prompt" in result_documents[0] and "ner_generated_text" in result_documents[0]:
        path_output_text = os.path.join(base_output_path, "prompt_and_response.txt")
        with open(path_output_text, "w") as f:
            for doc in result_documents:
                doc_key = doc["doc_key"]
                prompt = doc["ner_prompt"]
                generated_text = doc["ner_generated_text"]
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

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
