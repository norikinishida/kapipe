import argparse
import logging
import os

import sys
sys.path.insert(0, "../..")
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
    METHOD_NAME = args.METHOD_NAME

    # Input Data
    path_input_SOMETHING = args.input_SOMETHING

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
        "COMPONENT_NAME",
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    shared_functions.set_logger(
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
    SOMETHING = LOAD_FUNCTIONS(PATH_SOMETHING)
 
    ##################
    # Method
    ##################

    # Initialize the COMPONENT_NAME component
    if METHOD_NAME == "METHOD_A":
        WORKER = COMPONENT_WORKER_A()
    elif METHOD_NAME == "METHOD_B":
        WORKER = COMPONENT_WORKER_B()
    else:
        raise Exception(f"Invalid METHOD_NAME: {METHOD_NAME}")

    ##################
    # COMPONENT_NAME
    ##################

    logging.info(f"Applying the COMPONENT_NAME component to SOMETHING in {path_input_SOMETHING} ...")

    # Apply the COMPONENT_NAME component to the SOMETHING
    RESULTS = WORKER.WORK_SOMETHING(SOMETHING)

    # Save the COMPONENT_NAME results
    path_output_RESULTS = os.path.join(base_output_path, "RESULTS")
    SAVE_SOMETHING(path_output_RESULTS, RESULTS)
    logging.info(f"Saved RESULTS to {path_output_RESULTS}")

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
    parser.add_argument("--input_SOMETHING", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args) 
