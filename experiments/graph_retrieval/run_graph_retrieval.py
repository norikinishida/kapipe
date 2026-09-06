import argparse
import logging
import os
import sys

import networkx as nx
from tqdm import tqdm

from kapipe import utils
from kapipe.graph_retrieval import GraphRetriever
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
    input_anchor_contexts_path = args.input_anchor_contexts

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
        "graph_retrieval",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    set_logger(
        os.path.join(base_output_path, "graph_retrieval.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load graph
    logging.info("Loading graph ...")
    graph = nx.read_graphml(input_graph_path)
    logging.info(f"Number of nodes: {graph.number_of_nodes()}")
    logging.info(f"Number of edges: {graph.number_of_edges()}")

    # Load anchor contexts (queries)
    logging.info("Loading anchor contexts (queries) ...")
    anchor_contexts = utils.read_json(input_anchor_contexts_path)
    logging.info(f"Number of anchor contexts (queries): {len(anchor_contexts)}")

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

    # Instantiate the Graph Retrieval component
    retriever = GraphRetriever(
        use_timestamp=config["use_timestamp"],
    )

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the Graph Retrieval component to graph in {input_graph_path} with queries (anchors) in {input_anchor_contexts_path} ...")

    # Build the graph index
    retriever.make_index(graph=graph)

    # Apply the Graph Retrieval component to the graph
    for context_i, anchor_contexts_for_question in tqdm(
        enumerate(anchor_contexts),
        total=len(anchor_contexts)
    ):
        # Extract the anchor node IDs for the current question
        anchor_passages = anchor_contexts_for_question["contexts"]
        anchor_node_ids: list[str] = [
            passage[config["node_id_key"]]
            for passage in anchor_passages
        ]

        # Retrieve neighborhood nodes and edges based on the anchor passages
        nodes, edges = retriever.search(
            anchor_node_ids=anchor_node_ids,
            hop_size=config["hop_size"],
        )

        # Replace the anchor passages with the retrieved subgraph
        anchor_contexts[context_i]["contexts"] = None
        anchor_contexts[context_i]["nodes"] = nodes
        anchor_contexts[context_i]["edges"] = edges

    # Save the Graph Retrieval results
    base_filename = os.path.splitext(os.path.basename(input_anchor_contexts_path))[0]
    output_contexts_path = os.path.join(
        base_output_path,
        base_filename + ".graph_contexts.json"
    )
    utils.write_json(output_contexts_path, anchor_contexts)
    logging.info(f"Saved Graph Retrieval results to {output_contexts_path}")

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
    parser.add_argument("--input_anchor_contexts", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()
    main(args)
