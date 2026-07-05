import argparse
import logging
import os
import sys

import networkx as nx
import numpy as np

from kapipe import utils
from kapipe.community_clustering import (
    HierarchicalLeiden,
    NeighborhoodAggregation,
    TripleLevelFactorization
)
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
    input_graph_path = args.input_graph

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
        "community_clustering",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    set_logger(
        os.path.join(base_output_path, "community_clustering.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load knowledge graph
    logging.info("Loading knowledge graph ...")
    graph = nx.read_graphml(input_graph_path)

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the Community Clustering component
    if method_name == "hierarchical_leiden":
        # Instantiate the Hierarchical Leiden clusterer
        clusterer = HierarchicalLeiden(
            max_cluster_size=config["max_cluster_size"],
            use_lcc=config["use_lcc"]
        )
    elif method_name == "neighborhood_aggregation":
        # Instantiate the Neighborhood Aggregation clusterer
        clusterer = NeighborhoodAggregation(
            hop_size=config["hop_size"],
        )
    elif method_name == "triple_level_factorization":
        # Instantiate the Triple Level Factorization clusterer
        clusterer = TripleLevelFactorization()
    else:
        raise Exception(f"Invalid method_name: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the Community Clustering component to knowledge graph in {input_graph_path} ...")

    # Apply the Community Clustering component to the graph
    communities = clusterer.cluster_communities(graph=graph)

    # Save the results
    output_communities_path = os.path.join(base_output_path, "communities.json")
    utils.write_json(output_communities_path, communities)
    logging.info(f"Saved communities to {output_communities_path}")

    # Show statistics
    cluster_size_list = []
    level_list = []
    assert communities[0]["community_id"] == "ROOT"
    for community in communities[1:]:
        cluster_size_list.append(len(community["nodes"]))
        level_list.append(community["level"])
    logging.info(f"Number of Communities (with the Root Community): {len(communities)}")
    logging.info(f"Cluster Size Max: {np.max(cluster_size_list)}, Min: {np.min(cluster_size_list)}, Avg: {np.mean(cluster_size_list)}")
    logging.info(f"Level Max: {np.max(level_list)}, Min: {np.min(level_list)}")

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
    parser.add_argument("--input_graph", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()
    main(args) 
