import argparse
import logging
import os

import networkx as nx
import numpy as np

import sys
sys.path.insert(0, "../..")
from kapipe.community_clustering import (
    HierarchicalLeiden,
    NeighborhoodAggregation,
    TripleLevelFactorization
)
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    clustering_method = args.method

    # Input Data
    path_input_graph = args.input_graph

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
        "community_clustering",
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
        base_output_path + "/community_clustering.log",
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    logging.info("Loading knowledge graph ...")
    graph = nx.read_graphml(path_input_graph)

    ##################
    # Method
    ##################

    # Initialize the community clustering object
    if clustering_method == "hierarchical_leiden":
        clusterer = HierarchicalLeiden()
    elif clustering_method == "neighborhood_aggregation":
        clusterer = NeighborhoodAggregation()
    elif clustering_method == "triple_level_factorization":
        clusterer = TripleLevelFactorization()
    else:
        raise Exception(f"Invalid clustering_method: {clustering_method}")

    ##################
    # Community Clustering
    ##################

    # Apply the community clustering object to the graph
    logging.info("Clustering communities ...")
    if clustering_method == "hierarchical_leiden":
        communities = clusterer.cluster_communities(graph=graph, max_cluster_size=10, use_lcc=False)
        # communities = clusterer.cluster_communities(graph=graph, max_cluster_size=10, use_lcc=True)
    else:
        communities = clusterer.cluster_communities(graph=graph)

    # Save the communities
    utils.write_json(os.path.join(base_output_path, "communities.json"), communities)
    logging.info(f"Saved communities to {os.path.join(base_output_path, 'communities.json')}")

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

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()
    main(args) 
