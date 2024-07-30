import argparse
import logging
import os

import torch
import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
import kapipe
from kapipe import evaluation
from kapipe import utils


def main(args):
    torch.autograd.set_detect_anomaly(True)
    transformers.logging.set_verbosity_error()

    ka = kapipe.load(args.identifier)
        
    prefix = utils.get_current_time()

    utils.mkdir(os.path.join(args.results_dir, "pipeline", args.identifier, prefix))
    predict_and_eval(
        ka=ka,
        path_in_file=args.dev,
        path_out_file=os.path.join(
            args.results_dir,
            "pipeline",
            args.identifier,
            prefix,
            "dev"
        )
    )
    predict_and_eval(
        ka=ka,
        path_in_file=args.test,
        path_out_file=os.path.join(
            args.results_dir,
            "pipeline",
            args.identifier,
            prefix,
            "test"
        )
    )

def predict_and_eval(ka, path_in_file, path_out_file):
    # Read documents
    documents = utils.read_json(path_in_file)

    # Apply the pipeline to the documents
    logging.info(f"Applying the pipeline to {len(documents)} documents in {path_in_file} ...")
    preds = []
    for document in tqdm(documents):
        document = ka(document)
        preds.append(document)

    # Save the prediction results
    utils.write_json(path_out_file + ".pred.json", preds)
    logging.info(f"Saved the prediction results to {path_out_file + '.pred.json'}")

    # Evaluate the prediction results
    logging.info("Evaluating the prediction results ...")
    ner_scores = evaluation.ner.fscore(
        pred_path=path_out_file + ".pred.json",
        gold_path=path_in_file,
    )
    ed_scores = evaluation.ed.fscore(
        pred_path=path_out_file + ".pred.json",
        gold_path=path_in_file,
        inkb=False,
        skip_normalization=True,
        on_predicted_spans=True
    )
    docre_scores = evaluation.docre.fscore(
        pred_path=path_out_file + ".pred.json",
        gold_path=path_in_file,
        skip_intra_inter=True,
        skip_ign=True
    )
    scores = {
        "ner": ner_scores,
        "ed": ed_scores,
        "docre": docre_scores,
    }
    logging.info(utils.pretty_format_dict(scores))

    # Save the evaluation results
    utils.write_json(path_out_file + ".eval.json", scores)
    logging.info(f"Saved the evaluation results to {path_out_file + '.eval.json'}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
