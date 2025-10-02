import argparse
import logging
import os

import torch
import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.pipelines import TripleExtractionPipeline
from kapipe import (
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

    # Input Data
    path_input_documents = args.input_documents

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
        "pipelines",
        "triple_extraction_pipeline",
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

    pipe = TripleExtractionPipeline(
        module_kwargs={
            "chunking": {"model_name": "en_core_sci_md"},
            "ner": {"identifier": "gpt4omini_cdr"},
            "ed_retrieval": {"identifier": "dummy_entity_retriever"},
            "ed_reranking": {"identifier": "identical_entity_reranker"},
            "docre": {"identifier": "gpt4omini_cdr"}
        },
        share_backborn_llm=False
        #
        # module_kwargs={
        #     "chunking": {"model_name": "en_core_sci_md"},
        #     "ner": {"identifier": "biaffine_ner_cdr", "gpu": 1},
        #     "ed_retrieval": {"identifier": "blink_bi_encoder_cdr", "gpu": 1},
        #     "ed_reranking": {"identifier": "blink_cross_encoder_cdr", "gpu": 2},
        #     "docre": {"identifier": "atlop_cdr", "gpu": 3}
        # },
        # share_backborn_llm=False
        #
        # module_kwargs={
        #     "chunking": {"model_name": "en_core_sci_md"},
        #     "ner": {"identifier": "gpt4omini_cdr"},
        #     "ed_retrieval": {"identifier": "blink_bi_encoder_cdr", "gpu": 0},
        #     "ed_reranking": {"identifier": "gpt4omini_cdr"},
        #     "docre": {"identifier": "gpt4omini_cdr"}
        # },
        # share_backborn_llm=True
    )

    ##################
    # Triple Extraction
    ##################

    logging.info(f"Applying the Triple Extraction Pipeline to {len(documents)} documents in {path_input_documents} ...")

    # Create the full output path
    path_output_documents = os.path.join(base_output_path, "documents.json")

    # Apply the triple extraction pipeline to the documents
    result_documents = []
    for document in tqdm(documents):
        result_document = pipe.extract(document=document, num_candidate_entities=10)
        result_documents.append(result_document)
        if len(result_documents) % 500 == 0:
            utils.write_json(path_output_documents.replace(".json", f".until_{len(result_documents)}.json"), result_documents)

    # Save the results
    utils.write_json(path_output_documents, result_documents)
    logging.info(f"Saved the prediction results to {path_output_documents}")

    # Save the prompt-response pairs visually in plain text
    if "ner_prompt" in result_documents[0] and "ner_generated_text" in result_documents[0]:
        path_output_text = os.path.join(base_output_path, "prompt_and_response_ner.txt")
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
    if "ed_prompt" in result_documents[0] and "ed_generated_text" in result_documents[0]:
        path_output_text = os.path.join(base_output_path, "prompt_and_response_ed_reranking.txt")
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
    if "docre_prompt" in result_documents[0] and "docre_generated_text" in result_documents[0]:
        path_output_text = os.path.join(base_output_path, "prompt_and_response_docre.txt")
        with open(path_output_text, "w") as f:
            for doc in result_documents:
                doc_key = doc["doc_key"]
                prompt = doc["docre_prompt"]
                generated_text = doc["docre_generated_text"]
                f.write("-------------------------------------\n\n")
                f.write(f"DOC_KEY: {doc_key}\n\n")
                f.write("PROMPT:\n")
                f.write(prompt + "\n\n")
                f.write("GENERATED TEXT:\n")
                f.write(generated_text + "\n\n")
                f.flush()

    ##################
    # Evaluation
    ##################

    if do_evaluation:
        # Evaluate the prediction results
        logging.info("Evaluating the prediction results ...")
        ner_scores = evaluation.ner.fscore(
            pred_path=path_output_documents,
            gold_path=path_input_documents
        )
        ed_scores_mention_level = evaluation.ed.fscore(
            pred_path=path_output_documents,
            gold_path=path_input_documents,
            inkb=False,
            skip_normalization=True,
            on_predicted_spans=True
        )
        ed_scores_entity_level = evaluation.ed.entity_level_fscore(
            pred_path=path_output_documents,
            gold_path=path_input_documents
        )
        docre_scores = evaluation.docre.fscore(
            pred_path=path_output_documents,
            gold_path=path_input_documents,
            skip_intra_inter=True,
            skip_ign=True
        )
        scores = {
            "ner": ner_scores,
            "ed_mention_level": ed_scores_mention_level,
            "ed_entity_level": ed_scores_entity_level,
            "docre": docre_scores,
        }
        logging.info(utils.pretty_format_dict(scores))

        # Save the evaluation results
        path_eval_scores = os.path.join(base_output_path, "eval.json")
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

    # Input Data
    parser.add_argument("--input_documents", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Action
    parser.add_argument("--do_evaluation", action="store_true")

    args = parser.parse_args()

    main(args)
