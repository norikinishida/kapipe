import argparse
import logging
import os
import sys

from tqdm import tqdm

from kapipe import utils
from kapipe.llms import HuggingFaceLLM, OpenAILLM
from kapipe.proposition_relation_refinement import LLMPropositionRelationRefiner
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
    input_triples_path = args.input_triples

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    assert method_name in ["llm_proposition_relation_refiner"], f"Unknown method: {method_name}"

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "proposition_relation_refinement",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    set_logger(
        os.path.join(base_output_path, "proposition_relation_refinement.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load triples
    logging.info("Loading triples ...")
    triples = utils.read_json(input_triples_path)
    logging.info(f"Number of triples: {len(triples)}")

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

    # Instantiate the Proposition Relation Refinement component
    if method_name == "llm_proposition_relation_refiner":
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

        # Instantiate the LLM-based Proposition Relation Refinement component
        refiner = LLMPropositionRelationRefiner(
            model=model,
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            use_timestamp=config["use_timestamp"],
        )
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the Proposition Relation Refinement component to {len(triples)} triples in {input_triples_path} ...")

    # Apply the Proposition Relation Refinement component to the triples
    refined_triples = []
    n_deleted = 0
    for triple in tqdm(triples):
        refined_triple = refiner.refine(triple=triple)

        # Remove triples classified as NOREL
        if refined_triple["relation"] == "NOREL":
            n_deleted += 1
            continue

        refined_triples.append(refined_triple)

    logging.info(f"Refinement complete: {len(refined_triples)} triples kept, {n_deleted} triples removed")

    # Save the Proposition Relation Refinement results
    output_triples_path = os.path.join(base_output_path, "refined_triples.json")
    utils.write_json(output_triples_path, refined_triples)
    logging.info(f"Saved refined triples to {output_triples_path}")

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
    parser.add_argument("--input_triples", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()
    main(args)
