import argparse
import os

from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    utils.mkdir(os.path.dirname(args.output_file))
    entities = utils.read_json(args.input_file)
    results = []
    for e_i, e_page in enumerate(tqdm(entities)):
        entity_id = e_page["Class ID"]
        preferred_labels = e_page["Preferred Labels"]
        new_page = {
            "entity_id": entity_id,
            "entity_index": e_i,
            "entity_type": "UNKNOWN",
            "canonical_name": preferred_labels[0],
            "synonyms": preferred_labels[1:] + e_page["Synonyms"],
            "description": e_page["Definitions"],
            "tree_numbers": e_page["TreeNumbers"],
        }
        results.append(new_page)
    utils.write_json(args.output_file, results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args=args)