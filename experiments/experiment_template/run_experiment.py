import argparse
import logging
import os
import sys

from kapipe import utils
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
    INPUT_SOMETHING_PATH = args.input_SOMETHING

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
        "COMPONENT_NAME",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    set_logger(
        os.path.join(base_output_path, "COMPONENT_NAME.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load SOMETHING
    logging.info("Loading SOMETHING ...")
    SOMETHING = LOAD_FUNCTIONS(INPUT_SOMETHING_PATH)
 
    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the COMPONENT_NAME component
    if method_name == "default":
        WORKER = COMPONENT_WORKER_A(hoge=config["hoge"])
    else:
        raise Exception(f"Unknown method: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the COMPONENT_NAME component to SOMETHING in {INPUT_SOMETHING_PATH} ...")

    # Apply the COMPONENT_NAME component to the SOMETHING
    RESULTS = WORKER.WORK_SOMETHING(SOMETHING)

    # Save the results
    OUTPUT_RESULTS_PATH = os.path.join(base_output_path, "RESULTS")
    SAVE_SOMETHING(OUTPUT_RESULTS_PATH, RESULTS)
    logging.info(f"Saved the results to {OUTPUT_RESULTS_PATH}")

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
    parser.add_argument("--input_SOMETHING", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args) 
