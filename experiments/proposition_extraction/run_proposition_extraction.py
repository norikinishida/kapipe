import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

from kapipe import utils
from kapipe.llms import HuggingFaceLLM, OpenAILLM
from kapipe.proposition_extraction import LLMPropositionExtractor
from kapipe.utils import StopWatch


def main(args):
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
    input_passages_path = args.input_passages

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    assert method_name in ["llm_proposition_extractor"], f"Unknown method: {method_name}"

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "proposition_extraction",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    set_logger(
        os.path.join(base_output_path, "proposition_extraction.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name
    )

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    if method_name == "llm_proposition_extractor":
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

        # Instantiate the LLM-based Proposition Extraction component
        extractor = LLMPropositionExtractor(
            model=model,
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
        )
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    ##################
    # Method Execution
    ##################

    # Count the input passages
    with open(input_passages_path) as fin:
        n_lines = sum(1 for _ in fin)
    logging.info(f"Applying the Proposition Extraction component to {n_lines} passages in {input_passages_path} ...")

    # Create the output file path
    base_filename = os.path.splitext(os.path.basename(input_passages_path))[0]
    output_file_name = f"{base_filename}.propositions.jsonl"
    output_file_path = os.path.join(base_output_path, output_file_name)

    # Apply the Proposition Extraction component to the passages
    n_input_passages = 0
    n_output_propositions = 0
    with open(output_file_path, "w") as fout:
        with open(input_passages_path) as fin:
            for line in tqdm(fin, total=n_lines):
                # Load the passage
                passage = json.loads(line.strip())

                # Extract propositions from the passage
                propositions = extractor.extract(passage=passage)

                # Save the propositions
                for proposition in propositions:
                    json_str = json.dumps(proposition)
                    fout.write(json_str + "\n")

                # Count the processed passages and extracted propositions
                n_input_passages += 1
                n_output_propositions += len(propositions)

    logging.info(
        f"Extracted {n_output_propositions} propositions "
        f"from {n_input_passages} passages"
    )
    logging.info(f"Saved the propositions to {output_file_path}")

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))


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
    parser.add_argument("--input_passages", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()
    main(args) 
