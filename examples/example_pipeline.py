import argparse
import logging
import os

import kapipe
from kapipe import utils


def main(args):
    ka = kapipe.load(identifier="cdr_biaffinener_blink_atlop")

    document = utils.read_json(args.document)

    document = ka(document)

    path_output = os.path.splitext(args.document)[0]
    utils.write_json(path_output + ".pipe.json", document)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=str, required=True)
    args = parser.parse_args()

    main(args)
