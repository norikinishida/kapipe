import argparse
import logging
import os
import sys

# import numpy as np
from tqdm import tqdm
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.ed_reranking import IdenticalEntityReranker, BlinkCrossEncoder, LLMED
from kapipe.llms import HuggingFaceLLM, OpenAILLM
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
    input_candidate_entities_path = args.input_candidate_entities

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
        "ed_reranking",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Extract the base filename
    base_filename = os.path.splitext(os.path.basename(input_documents_path))[0]

    # Set logger
    set_logger(
        os.path.join(base_output_path, f"{base_filename}.ed_reranking.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load documents
    documents = utils.read_json(input_documents_path)

    # Load candidate entities
    candidate_entities = utils.read_json(input_candidate_entities_path)

    # Check that the documents and candidate entities match
    assert len(documents) == len(candidate_entities), f"Number of documents and candidate entities do not match: {len(documents)} vs {len(candidate_entities)}"
    for doc, cands in zip(documents, candidate_entities):
        assert doc["doc_key"] == cands["doc_key"], f"Document and candidate entities do not match: {doc['doc_key']} vs {cands['doc_key']}"

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the ED-Reranking component
    if method_name == "identical_entity_reranker":
        # Instantiate the ED-Reranking component using identical function
        reranker = IdenticalEntityReranker()
    elif method_name == "blink_cross_encoder":
        # Load the BLINK Cross-Encoder ED-Reranking component from the public snapshot
        reranker = BlinkCrossEncoder.from_identifier(
            identifier=config["identifier"]
        )
    elif method_name == "llm_ed":
        # Instantiate the LLM wrapper
        if config["llm_provider"] == "openai":
            model = OpenAILLM(
                model_name=config["llm_model_name"],
                max_new_tokens=config["llm_max_new_tokens"],
            )
        elif config["llm_provider"] == "hf":
            model = HuggingFaceLLM(
                model_name=config["llm_model_name"],
                max_new_tokens=config["llm_max_new_tokens"],
                quantization_bits=config["llm_quantization_bits"],
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config['llm_provider']}")
        logging.info("Instantiated the LLM model: %s" % repr(model))

        # Load the LLM-based ED-Reranking component from the public snapshot
        reranker = LLMED.from_identifier(
            model=model,
            identifier=config["identifier"]
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    ##################
    # Method Execution
    ##################
    
    logging.info(f"Applying the ED-Reranking component to {len(documents)} documents (+ candidate entities) in {input_documents_path} ({input_candidate_entities_path}) ...")

    # Apply the ED-Reranking component to the documents (with candidate entities)
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

    # Save the results
    output_documents_path = os.path.join(
        base_output_path,
        f"{base_filename}.pred.json"
    )
    utils.write_json(output_documents_path, result_documents)
    logging.info(f"Saved the prediction results to {output_documents_path}")

    # Save the prompt-response pairs visually in plain text
    if (
        len(result_documents) > 0
        and "ed_prompt" in result_documents[0]
        and "ed_generated_text" in result_documents[0]
    ):
        output_text_path = os.path.join(
            base_output_path,
            f"{base_filename}.prompt_and_response.txt"
        )
        with open(output_text_path, "w") as f:
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
        if hasattr(reranker, "entity_dict"):
            # Use the entity dictionary bundled in the reranker component
            kb_entity_ids = set(reranker.entity_dict.keys())

        # Enable InKB evaluation only when the reranker exposes its entity dictionary
        inkb = kb_entity_ids is not None

        for gold_doc, candidate_entities_for_doc in zip(
            gold_documents,
            candidate_entities
        ):
            # Check document alignment
            assert gold_doc["doc_key"] == candidate_entities_for_doc["doc_key"]

            for gold_mention, candidates_for_mention in zip(
                gold_doc["mentions"],
                candidate_entities_for_doc["candidate_entities"],
            ):

                # Mark whether the gold entity exists in the entity dictionary
                if kb_entity_ids is not None:
                    gold_mention["in_kb"] = (
                        gold_mention["entity_id"] in kb_entity_ids
                    )

                # Collect candidate entity IDs for this mention
                candidate_entity_ids = [
                    candidate["entity_id"]
                    for candidate in candidates_for_mention
                ]

                # Mark whether the gold entity is included in candidates
                gold_mention["in_cand"] = (
                    gold_mention["entity_id"] in candidate_entity_ids
                )

        # Evaluate the prediction results
        scores = evaluation.ed.accuracy(
            pred_path=output_documents_path,
            gold_path=gold_documents,
            inkb=inkb,
            skip_normalization=False,
        )
        scores.update(
            evaluation.ed.fscore(
                pred_path=output_documents_path,
                gold_path=gold_documents,
                inkb=inkb,
                skip_normalization=False,
            )
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
    parser.add_argument("--input_candidate_entities", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
