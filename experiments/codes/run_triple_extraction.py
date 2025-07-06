import argparse
import logging
import os

import torch
import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe import (
    triple_extraction,
    evaluation,
    utils
)
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    torch.autograd.set_detect_anomaly(True)
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    identifier = args.identifier

    # Input Data
    path_input_documents_list = args.input_documents_list

    # Output Path
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    do_evaluation = args.do_evaluation

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "triple_extraction",
        "pipeline",
        identifier,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(base_output_path + "/extraction.log")

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load documents
    documents = []
    for path_input_documents in path_input_documents_list:
        docs = utils.read_json(path_input_documents)
        documents.extend(docs)

    ##################
    # Method
    ##################

    pipe = triple_extraction.load(
        identifier=identifier,
        gpu_map={"ner": 0, "ed_retrieval": 0, "ed_reranking": 2, "docre": 3}
    )

    ##################
    # Triple Extraction
    ##################

    # Create the full output path
    filename = os.path.splitext(os.path.basename(path_input_documents_list[0]))[0]
    path_output_documents = os.path.join(base_output_path, f"{filename}.pred.json")

    # Apply the pipeline to the documents
    logging.info(f"Applying the pipeline to {len(documents)} documents in {path_input_documents_list} ...")
    preds = []
    for document in tqdm(documents):
        document = pipe(document)
        preds.append(document)
        if len(preds) % 500 == 0:
            utils.write_json(path_output_documents.replace(".pred.json", f".pred_until_{len(preds)}.json"), preds)

    # Save the prediction results
    utils.write_json(path_output_documents, preds)
    logging.info(f"Saved the prediction results to {path_output_documents}")

    ##################
    # Evaluation
    ##################

    if do_evaluation:
        assert len(path_input_documents_list) == 1

        # Evaluate the prediction results
        logging.info("Evaluating the prediction results ...")
        ner_scores = kapipe.evaluation.ner.fscore(
            pred_path=path_output_documents,
            gold_path=path_input_documents_list[0]
        )
        ed_scores = kapipe.evaluation.ed.fscore(
            pred_path=path_output_documents,
            gold_path=path_input_documents_list[0],
            inkb=False,
            skip_normalization=True,
            on_predicted_spans=True
        )
        ed_scores2 = kapipe.evaluation.ed.entity_level_fscore(
            pred_path=path_output_documents,
            gold_path=path_input_documents_list[0]
        )
        docre_scores = kapipe.evaluation.docre.fscore(
            pred_path=path_output_documents,
            gold_path=path_input_documents_list[0],
            skip_intra_inter=True,
            skip_ign=True
        )
        scores = {
            "ner": ner_scores,
            "ed": ed_scores,
            "ed_entity_level": ed_scores2,
            "docre": docre_scores,
        }
        logging.info(utils.pretty_format_dict(scores))

        # Save the evaluation results
        path_eval_scores = os.path.join(base_output_path, f"{filename}.eval.json")
        utils.write_json(path_eval_scores, scores)
        logging.info(f"Saved the evaluation results to {path_eval_scores}")

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
    parser.add_argument("--identifier", type=str, required=True)

    # Input Data
    parser.add_argument("--input_documents_list", nargs="+")

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Action
    parser.add_argument("--do_evaluation", action="store_true")

    args = parser.parse_args()

    main(args)
