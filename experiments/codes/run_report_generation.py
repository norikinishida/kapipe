import argparse
import json
import logging
import os

import networkx as nx
import torch
import transformers

import sys
sys.path.insert(0, "../..")
from kapipe.report_generation import (
    LLMBasedReportGenerator,
    TemplateBasedReportGenerator
)
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


RELATION_MAP = {
    "CID": "Chemical-Induce-Disease"
}


def main(args):
    torch.autograd.set_detect_anomaly(True)
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    reporting_method = args.method

    # Input Data
    path_input_graph = args.input_graph
    path_input_communities = args.input_communities
    node_attr_keys = args.node_attr_keys
    edge_attr_keys = args.edge_attr_keys

    # Output Path
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix
 
    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "report_generation",
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
        os.path.join(base_output_path, "report_generation.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load the knowledge graph
    logging.info("Loading knowledge graph ...")
    graph = nx.read_graphml(path_input_graph)

    # Load the community records
    logging.info("Loading community records ...")
    communities = utils.read_json(path_input_communities)

    ##################
    # Method
    ##################

    # Initialize the Report Generation component
    if reporting_method == "llm":
        generator = LLMBasedReportGenerator(
            llm_backend="openai",
            llm_kwargs={
                "openai_model_name": "gpt-4o-mini",
                "max_new_tokens": 2048
            }
        )
        # OR
        # generator = LLMBasedReportGenerator(
        #     llm_backend="huggingface",
        #     llm_kwargs={
        #         "llm_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        #         "max_new_tokens": 2048,
        #         "quantization_bits": -1,
        #     }
        # )
    elif reporting_method == "template":
        generator = TemplateBasedReportGenerator()
    else:
        raise Exception(f"Invalid reporting_method: {reporting_method}")

    ##################
    # Report Generation
    ##################

    logging.info(f"Applying the Report Generation component to {len(communities)} communities in {path_input_communities} ...")

    # Apply the report generator to the communities
    reports = generator.generate_community_reports(
        # Input
        graph=graph,
        communities=communities,
        node_attr_keys=node_attr_keys,
        edge_attr_keys=edge_attr_keys,
        # Misc.
        relation_map=RELATION_MAP
    )

    # Save the Report Generation results
    path_output_reports = os.path.join(base_output_path, "reports.jsonl")
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


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True)

    # Input Data
    parser.add_argument("--input_graph", type=str, required=True)
    parser.add_argument("--input_communities", type=str, required=True)
    parser.add_argument("--node_attr_keys", nargs="+", default=["name", "entity_type", "description"])
    parser.add_argument("--edge_attr_keys", nargs="+", default=["relation"])

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args) 
