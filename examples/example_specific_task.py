import argparse
import logging
import os

import kapipe
from kapipe import utils


def main(args):
    ka = kapipe.load(identifier="cdr_biaffinener_blink_atlop")

    document = utils.read_json(args.document)
    path_output = os.path.splitext(args.document)[0]

    document = ka.ner(document)
    utils.write_json(path_output + ".ner.json", document)

    document, candidate_entities = ka.ed_ret(document)
    utils.write_json(path_output + ".ner.ed_ret.json", document)

    document = ka.ed_rank(document, candidate_entities)
    utils.write_json(path_output + ".ner.ed_ret.ed_rank.json", document)

    document = ka.docre(document)
    utils.write_json(path_output + ".ner.ed_ret.ed_rank.docre.json", document)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=str, required=True)
    args = parser.parse_args()

    main(args)
