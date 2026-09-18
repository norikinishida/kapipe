import argparse
import logging
import os
import sys

from tqdm import tqdm
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.llms import HuggingFaceLLM, OpenAILLM
from kapipe.ner import BiaffineNER, LLMNER
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
        "ner",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Extract the base filename
    base_filename = os.path.splitext(os.path.basename(input_documents_path))[0]

    # Set logger
    set_logger(
        os.path.join(base_output_path, f"{base_filename}.ner.log"),
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

    # Instantiate the NER component
    if method_name == "biaffine_ner":
        # Load the Biaffine NER component from the public snapshot
        extractor = BiaffineNER.from_identifier(
            identifier=config["identifier"]
        )
    elif method_name == "llm_ner":
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

        if "identifier" in config:
            # Load the LLM-based NER component from the public snapshot
            extractor = LLMNER.from_identifier(
                model=model,
                identifier=config["identifier"]
            )
        else:
            # Load the user-defined schema
            vocab_etype: dict[str, int] = {
                etype: etype_i
                for etype_i, etype in enumerate(config["entity_types"])
            }
            etype_meta_info: dict[str, dict[str, str]] = config["etype_meta_info"]

            # Instantiate the LLM-based NER component with the user-defined schema
            extractor = LLMNER(
                model=model,
                prompt_template_name_or_path=config["prompt_template_name_or_path"],
                vocab_etype=vocab_etype,
                etype_meta_info=etype_meta_info,
            )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the NER component to {len(documents)} documents in {input_documents_path} ...")

    # Apply the NER component to the documents
    result_documents = []
    for document in tqdm(documents):
        result_document = extractor.extract(document=document)
        result_documents.append(result_document)

    # Save the results
    output_documents_path = os.path.join(
        base_output_path,
        f"{base_filename}.pred.json"
    )
    utils.write_json(output_documents_path, result_documents)
    logging.info(f"Saved the prediction results to {output_documents_path}")

    # Save the prompt-response pairs visually in plain text
    if "ner_prompt" in result_documents[0] and "ner_generated_text" in result_documents[0]:
        output_text_path = os.path.join(
            base_output_path,
            f"{base_filename}.prompt_and_response.txt"
        )
        with open(output_text_path, "w") as f:
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
    # Evaluation
    ##################

    if do_evaluation:
        # Validate that the gold documents path is provided
        if gold_documents_path is None:
            raise ValueError("--gold is required when --do_evaluation is set")

        # Evaluate the prediction results
        scores = evaluation.ner.fscore(
            pred_path=output_documents_path,
            gold_path=gold_documents_path,
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

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
