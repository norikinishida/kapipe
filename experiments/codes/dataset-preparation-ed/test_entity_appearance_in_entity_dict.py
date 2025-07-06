import argparse

from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    print("Loading documents ...")
    documents = utils.read_json(args.input_file)
    print(f"Loaded {len(documents)} documents")

    print("Loading entity dictionary ...")
    entity_dict = utils.read_json(args.entity_dict) # list[EntityPage]
    entity_dict = {epage["entity_id"]: epage for epage in entity_dict} # dict[str, EntityPage]
    print(f"Found {len(entity_dict)} pages in the loaded entity dictionary")

    not_found_entities = []
    found_entities = []
    for document in tqdm(documents):
        for entity in document["entities"]:
            if not entity["entity_id"] in entity_dict:
                not_found_entities.append(entity["entity_id"])
            else:
                found_entities.append(entity["entity_id"])

    not_found_entities = sorted(list(set(not_found_entities)))
    found_entities = sorted(list(set(found_entities)))
    print(f"{len(not_found_entities)} entities were NOT found in the entity dictionary.")
    print(f"{len(found_entities)} entities were found in the entity dictionary.")
    with open("not_found_entities.txt", "w") as f:
        for i, x in enumerate(not_found_entities):
            f.write(f"[{i}] {x}\n")
    print(f"Saved the output to not_found_entities.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, required=True)
    args = parser.parse_args()
    main(args)