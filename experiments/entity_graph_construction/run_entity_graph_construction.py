import argparse
import logging
import os
import sys

import networkx as nx

from kapipe import utils
from kapipe.entity_graph_construction import EntityGraphConstructor
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
    input_documents_path_list = args.input_documents_list
    input_additional_triples_path = args.input_additional_triples
    input_entity_dict_path = args.input_entity_dict

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
        "entity_graph_construction",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    set_logger(
        os.path.join(base_output_path, "entity_graph_construction.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Initialize the Entity Graph Construction component
    constructor = EntityGraphConstructor(
        missing_entity_policy=config["missing_entity_policy"],
        missing_entity_description=config["missing_entity_description"],
    )

    ##################
    # Method Execution
    ##################
    
    logging.info(f"Applying the Entity Graph Construction component to extracted triples ({input_documents_path_list}) and additional triples ({input_additional_triples_path}) ...")

    # Apply the Entity Graph Construction component to extracted triples and
    # additional triples (optional).
    # The entity dictionary is used to label canonical names, synonyms, entity types,
    # and definitions to each node as their attributes.
    graph = constructor.construct_entity_graph(
        documents_path_list=input_documents_path_list,
        entity_dict_path=input_entity_dict_path,
        additional_triples_path=input_additional_triples_path,
    )

    # Save the `networkx.MultiDiGraph` in GraphML format
    output_graph_path = os.path.join(base_output_path, "graph.graphml")
    nx.write_graphml(graph, output_graph_path)
    logging.info(f"Saved graph to {output_graph_path}")

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
    parser.add_argument("--input_documents_list", nargs="*")
    parser.add_argument("--input_additional_triples", type=str, default=None)
    parser.add_argument("--input_entity_dict", type=str, default=None)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args) 
