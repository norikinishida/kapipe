import argparse

from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    documents = utils.read_json(args.input_file)
    entity_dict = utils.read_json(args.entity_dict) # list[EntityPage]
    entity_dict = {epage["entity_id"]: epage for epage in entity_dict} # dict[str, EntityPage]

    not_found_entities = []
    for document in tqdm(documents):
        for entity in document["entities"]:
            if not entity["entity_id"] in entity_dict:
                not_found_entities.append(entity["entity_id"])

    utils.print_list(not_found_entities)
    print(f"{len(not_found_entities)} entities were NOT found in the entity dictionary.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, required=True)
    args = parser.parse_args()
    main(args)