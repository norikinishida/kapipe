import argparse
import logging
import os

import networkx as nx

import sys
sys.path.insert(0, "../..")
from kapipe.graph_construction import GraphConstructor
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Input Data
    path_documents_list = args.documents_list
    path_additional_triples = args.additional_triples
    path_entity_dict = args.entity_dict

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
        "graph_construction",
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
        base_output_path + "/graph_construction.log",
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Method
    ##################

    constructor = GraphConstructor()

    ##################
    # Knowledge Graph Construction
    ##################

    # Apply the knowledge graph construction to the documents with extracted triples and additional triples (optional)
    # The entity dictionary is used to label canonical names, synonyms, entity types, and definitions to each node as their attributes
    graph = constructor.construct_knowledge_graph(
        path_documents_list=path_documents_list,
        path_additional_triples=path_additional_triples,
        path_entity_dict=path_entity_dict
    )

    # Save the `networkx.MultiDiGraph` in GraphML format
    nx.write_graphml(graph, os.path.join(base_output_path, "graph.graphml"))

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
    parser.add_argument("--documents_list", nargs="*")
    parser.add_argument("--additional_triples", type=str, default=None)
    parser.add_argument("--entity_dict", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args) 
