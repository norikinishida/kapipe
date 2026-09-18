import argparse
import logging
import os
import sys

from tqdm import tqdm

from kapipe import utils
from kapipe.context_formatting import GraphVerbalizer
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
    input_graph_contexts_path = args.input_graph_contexts

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
        "context_formatting",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Extract the base filename
    base_filename = os.path.splitext(os.path.basename(input_graph_contexts_path))[0]

    # Set logger
    set_logger(
        os.path.join(
            base_output_path,
            f"{base_filename}.context_formatting.log",
        ),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load graph contexts
    logging.info("Loading graph contexts ...")
    graph_contexts = utils.read_json(input_graph_contexts_path)
    logging.info(f"Number of graph contexts: {len(graph_contexts)}")

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

    # Instantiate the Context Formatting component
    if method_name == "graph_verbalizer":
        formatter = GraphVerbalizer(
            use_timestamp=config["use_timestamp"],
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(
        f"Applying the Context Formatting component to graph contexts "
        f"in {input_graph_contexts_path} ..."
    )

    # Apply the Context Formatting component to the graph contexts
    for context_i, graph_contexts_for_question in tqdm(
        enumerate(graph_contexts),
        total=len(graph_contexts)
    ):
        # Convert the graph into LLM-readable context
        text = formatter.convert(
            nodes=graph_contexts_for_question["nodes"],
            edges=graph_contexts_for_question["edges"],
        )

        # Store the formatted context in the ContextsForOneExample format
        graph_contexts[context_i]["contexts"] = [
            {
                "passage_key": (
                    f"{graph_contexts_for_question['question_key']}/context#0000"
                ),
                "text": text,
            }
        ]

    # Save the Context Formatting results
    output_contexts_path = os.path.join(
        base_output_path,
        f"{base_filename}.formatted_contexts.json"
    )
    utils.write_json(output_contexts_path, graph_contexts)
    logging.info(f"Saved Context Formatting results to {output_contexts_path}")

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
    parser.add_argument("--input_graph_contexts", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()
    main(args)
