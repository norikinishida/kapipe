import argparse
import logging
import os
import sys

from tqdm import tqdm
import transformers

from kapipe import utils
from kapipe.ed_retrieval import MentionNameEntityRetriever, BlinkBiEncoder
from kapipe.utils import StopWatch


def main(args):
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
    input_documents_path = args.input_documents

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    assert method_name in ["mention_name_entity_retriever", "blink_bi_encoder"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "ed_retrieval",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)
    
    # Set logger
    set_logger(
        os.path.join(base_output_path, "ed_retrieval.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load documents
    documents = utils.read_json(input_documents_path)

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the ED-Retrieval component.
    # Also build the index over entities.
    if method_name == "mention_name_entity_retriever":
        # Instantiate the Mention-Name Entity retriever
        retriever = MentionNameEntityRetriever()
        retriever.make_index()
    elif method_name == "blink_bi_encoder":
        # Load the BLINK Bi-Encoder retriever from the public snapshot
        retriever = BlinkBiEncoder.from_identifier(
            identifier=config["identifier"]
        )
        retriever.make_index(use_precomputed_entity_vectors=True)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the ED-Retrieval component to {len(documents)} documents in {input_documents_path} ...")

    # Create the full output path
    output_documents_path = os.path.join(
        base_output_path,
        "documents.json"
    )
    output_candidates_path = os.path.join(
        base_output_path,
        "candidate_entities.json"
    )

    # Apply the ED-Retrieval component to the documents
    result_documents = []
    candidate_entities = []
    for document in tqdm(documents):
        result_document, candidate_entities_for_doc = retriever.search(
            document=document,
            retrieval_size=config["retrieval_size"]
        )
        result_documents.append(result_document)
        candidate_entities.append(candidate_entities_for_doc)

    # Save the results
    utils.write_json(output_documents_path, result_documents)
    utils.write_json(output_candidates_path, candidate_entities)
    logging.info(f"Saved the prediction results to {output_documents_path} and {output_candidates_path}")

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


def set_logger(filename: str, overwrite: bool = False) -> None:
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
    parser.add_argument("--input_documents", type=str, required=True)

    # Output Data
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
