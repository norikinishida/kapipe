import argparse
import logging
import os
import sys

from tqdm import tqdm
import transformers

from kapipe import evaluation
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

    # Evaluation
    do_evaluation = args.do_evaluation
    gold_documents_path = args.gold

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
    
    base_filename = os.path.splitext(os.path.basename(input_documents_path))[0]

    # Set logger
    set_logger(
        os.path.join(base_output_path, f"{base_filename}.ed_retrieval.log"),
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
        # Instantiate the ED-Retrieval component using simple mention-name assignment
        retriever = MentionNameEntityRetriever()
        retriever.make_index()
    elif method_name == "blink_bi_encoder":
        # Load the BLINK Bi-Encoder ED-Retrieval component from the public snapshot
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
    output_documents_path = os.path.join(
        base_output_path,
        f"{base_filename}.pred.json"
    )
    utils.write_json(output_documents_path, result_documents)

    output_candidates_path = os.path.join(
        base_output_path,
        f"{base_filename}.pred_candidate_entities.json"
    )
    utils.write_json(output_candidates_path, candidate_entities)

    logging.info(f"Saved the prediction results to {output_documents_path} and {output_candidates_path}")

    ##################
    # Evaluation
    ##################

    if do_evaluation:
        # Validate that the gold documents path is provided
        if gold_documents_path is None:
            raise ValueError("--gold is required when --do_evaluation is set")

        # Load the gold documents to assign meta information for evaluation
        gold_documents = utils.read_json(gold_documents_path)

        # Read the entity dictionary from the instantiated reranker
        kb_entity_ids = None
        if hasattr(retriever, "entity_dict"):
            # Use the entity dictionary bundled in the reranker component
            kb_entity_ids = set(retriever.entity_dict.keys())

        # Enable InKB evaluation only when the reranker exposes its entity dictionary
        inkb = kb_entity_ids is not None

        if kb_entity_ids is not None:
            for gold_doc in gold_documents:
                for gold_mention in gold_doc["mentions"]:
                    # Mark whether the gold entity exists in the entity dictionary
                    gold_mention["in_kb"] = (
                        gold_mention["entity_id"] in kb_entity_ids
                    )

        # Evaluate the prediction results
        scores = evaluation.ed.recall_at_k(
            pred_path=output_candidates_path,
            gold_path=gold_documents,
            inkb=inkb
        )
        logging.info(utils.pretty_format_dict(scores))
 
        # Save the evaluation result
        output_evaluation_path = os.path.join(
            base_output_path,
            f"{base_filename}.eval.json"
        )
        utils.write_json(output_evaluation_path, scores)
        logging.info(f"Saved the evaluation results to {output_evaluation_path}")

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
    logging.getLogger("httpx").addFilter(
        lambda r: "huggingface.co" not in r.getMessage()
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

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
