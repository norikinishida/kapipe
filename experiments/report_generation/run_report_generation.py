import argparse
import json
import logging
import os
import sys

import networkx as nx
import torch
import transformers

from kapipe import utils
from kapipe.llms import HuggingFaceLLM, OpenAILLM
from kapipe.report_generation import (
    LLMBasedReportGenerator,
    TemplateBasedReportGenerator
)
from kapipe.utils import StopWatch


def main(args):
    torch.autograd.set_detect_anomaly(True)
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
    input_graph_path = args.input_graph
    input_communities_path = args.input_communities

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
        "report_generation",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Extract the base filename
    base_filename = os.path.splitext(os.path.basename(input_communities_path))[0]

    # Set logger
    set_logger(
        os.path.join(
            base_output_path,
            f"{base_filename}.report_generation.log",
        ),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load the knowledge graph
    logging.info("Loading knowledge graph ...")
    graph = nx.read_graphml(input_graph_path)

    # Load the community records
    logging.info("Loading community records ...")
    communities = utils.read_json(input_communities_path)

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)
 
    # Instantiate the Report Generation component
    if method_name == "llm_based_report_generator":
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

        # Instantiate the LLM-based Report Generation component
        generator = LLMBasedReportGenerator(
            model=model,
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            relation_map=config["relation_map"],
        )
    elif method_name == "template_based_report_generator":
        # Instantiate the template-based Report Generation component
        generator = TemplateBasedReportGenerator(
            relation_map=config["relation_map"],
        )
    else:
        raise Exception(f"Invalid method: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the Report Generation component to {len(communities)} communities in {input_communities_path} ...")

    # Apply the report generator to the communities
    reports = generator.generate_community_reports(
        # Input
        graph=graph,
        communities=communities,
        node_attr_keys=tuple(config["node_attr_keys"]),
        edge_attr_keys=tuple(config["edge_attr_keys"]),
    )

    # Save the Report Generation results
    path_output_reports = os.path.join(
        base_output_path,
        f"{base_filename}.reports.jsonl",
    )
    with open(path_output_reports, "w") as f:
        for r in reports:
            line = json.dumps(r)
            f.write(line + "\n")
    logging.info(f"Saved reports to {path_output_reports}")

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
    logging.getLogger("httpx").addFilter(
        lambda r: "huggingface.co" not in r.getMessage()
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--input_graph", type=str, required=True)
    parser.add_argument("--input_communities", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args) 
