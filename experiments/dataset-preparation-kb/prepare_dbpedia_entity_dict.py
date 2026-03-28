import argparse
import os
import re

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


rdf_regex = re.compile(
    r'<http:\/\/dbpedia\.org\/resource\/([^\s]+)> <[^\s]+> "(.*)"@(\w+) \.'
)


def process_rdf_line(rdf_line: str):
    result = rdf_regex.search(rdf_line)
    
    resource =result.group(1)
    language = result.group(3)
    
    abstract = result.group(2)
    abstract = abstract.replace("\\", "")

    return {
        "text": abstract,
        "url": resource,
        "language": language
    }


def main(args):
    path_input_file = args.input_file
    path_output_file = args.output_file

    utils.mkdir(os.path.dirname(path_output_file))

    print(f"Loading Wikipedia entities (articles) from {path_input_file} ...")
    entity_dict = {}
    with open(path_input_file) as f:
        for line in f:

            if line.startswith("#"):
                continue

            data = process_rdf_line(line)

            if data["language"] != "en":
                continue

            entity_id = data["url"]
            canonical_name = entity_id.replace("_", " ")
            description = data["text"]

            if entity_id in entity_dict:
                raise Exception(f"There are multiple articles with the same title: {entity_id}")

            entity_dict[entity_id] = {
                "entity_id": entity_id,
                "entity_index": len(entity_dict),
                "entity_type": None,
                "canonical_name": canonical_name,
                "synonyms": [],
                "description": description,
                "tree_numbers": []
            }
    print("Completed loading")
    print(f"Loaded {len(entity_dict)} Wikipedia entities (articles)")

    print("Convering to a single list of entity pages ...")
    entity_dict = list(entity_dict.values())                
    print("Completed converting")

    print(f"Saving the entity dictionary to {path_output_file} ...")
    utils.write_json(path_output_file, entity_dict)
    print("Completed saving")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

