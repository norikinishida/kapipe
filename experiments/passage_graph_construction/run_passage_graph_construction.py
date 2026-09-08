import argparse
import json
import logging
import os
import sys

import networkx as nx

from kapipe import utils
from kapipe.passage_graph_construction import PassageGraphConstructor
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
    input_triples_path = args.input_triples

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
        "passage_graph_construction",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    set_logger(
        os.path.join(base_output_path, "passage_graph_construction.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load passages
    logging.info("Loading passages ...")
    passages = []
    with open(input_passages_path) as f:
        for line in f:
            passage = json.loads(line.strip())
            passages.append(passage)
    logging.info(f"Number of passages: {len(passages)}")

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

    # Instantiate the Passage Graph Construction component
    constructor = PassageGraphConstructor()

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the Passage Graph Construction component to {len(triples)} triples in {input_triples_path} ...")

    # Apply the Passage Graph Construction component to the triples
    graph = constructor.construct_passage_graph(
        passages=passages,
        triples=triples,
    )

    # Show statistics
    show_graph_statistics(graph)

    # Save the Passage Graph Construction results
    output_graph_path = os.path.join(base_output_path, "graph.graphml")
    nx.write_graphml(graph, output_graph_path)
    logging.info(f"Saved graph to {output_graph_path}")

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))


def show_graph_statistics(graph: nx.Graph | nx.DiGraph) -> None:
    # Count nodes and edges
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Calculate graph density
    density = nx.density(graph)

    # Calculate in-degree and out-degree statistics
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    avg_in = sum(in_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_out = sum(out_degrees.values()) / num_nodes if num_nodes > 0 else 0
    max_in = max(in_degrees.values()) if num_nodes > 0 else 0
    max_out = max(out_degrees.values()) if num_nodes > 0 else 0

    # Count strongly and weakly connected components
    num_scc = nx.number_strongly_connected_components(graph)
    num_wcc = nx.number_weakly_connected_components(graph)

    # Organize graph statistics
    statistics = {
        "nodes": num_nodes,
        "edges": num_edges,
        "density": density,
        "directed": graph.is_directed(),
        "avg_in_degree": avg_in,
        "avg_out_degree": avg_out,
        "max_in_degree": max_in,
        "max_out_degree": max_out,
        "num_strongly_connected_components": num_scc,
        "num_weakly_connected_components": num_wcc,
    }

    # Show every graph statistic
    for key, value in statistics.items():
        logging.info(f"{key}: {value}")


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
    parser.add_argument("--input_triples", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()
    main(args)
