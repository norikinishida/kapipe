import argparse
import logging
import os
import sys

from tqdm import tqdm
import transformers

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
    identifier = args.identifier

    llm_provider = args.llm_provider
    llm_model_name = args.llm_model_name
    llm_max_new_tokens = args.llm_max_new_tokens
    llm_quantization_bits = args.llm_quantization_bits

    # Input Data
    input_documents_path = args.input_documents

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "ner",
        method_name,
        identifier,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    set_logger(
        os.path.join(base_output_path, "extraction.log"),
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
    # Method
    ##################

    # Initialize the NER extractor
    if method_name == "biaffine_ner":
        extractor = BiaffineNER.from_identifier(identifier=identifier)
    elif method_name == "llm_ner":
        if llm_provider == "openai":
            model = OpenAILLM(
                model_name=llm_model_name,
                max_new_tokens=llm_max_new_tokens,
            )
        elif llm_provider == "hf":
            model = HuggingFaceLLM(
                model_name=llm_model_name,
                max_new_tokens=llm_max_new_tokens,
                quantization_bits=llm_quantization_bits,
            )
        extractor = LLMNER.from_identifier(model=model, identifier=identifier)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    ##################
    # NER
    ##################

    logging.info(f"Applying the NER component to {len(documents)} documents in {input_documents_path} ...")

    # Create the full output path
    output_documents_path = os.path.join(base_output_path, "documents.json")

    # Apply the NER extractor to the documents
    result_documents = []
    for document in tqdm(documents):
        result_document = extractor.extract(document=document)
        result_documents.append(result_document)

    # Save the results
    utils.write_json(output_documents_path, result_documents)
    logging.info(f"Saved the prediction results to {output_documents_path}")

    # Save the prompt-response pairs visually in plain text
    if "ner_prompt" in result_documents[0] and "ner_generated_text" in result_documents[0]:
        output_text_path = os.path.join(base_output_path, "prompt_and_response.txt")
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
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


def set_logger(filename, overwrite=False):
    """
    Parameters
    ----------
    filename: str
    overwrite: bool, default False
    """
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
    parser.add_argument("--identifier", type=str, required=True)

    parser.add_argument("--llm_provider", type=str, default="openai")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_max_new_tokens", type=int, default=1024)
    parser.add_argument("--llm_quantization_bits", type=int, default=None)

    # Input Data
    parser.add_argument("--input_documents", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
