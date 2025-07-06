import argparse
import os

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils



def main(args):
    path_input_triples = args.input_triples
    path_entity_dict = args.entity_dict
    path_output_triples = args.output_triples

    utils.mkdir(os.path.dirname(path_output_triples))

    print(f"Loading triples from {path_input_triples}")
    triples = utils.read_json(path_input_triples)
    print(f"Loaded {len(triples)} triples")

    print(f"Loading entity dictionary from {path_entity_dict}")
    entity_dict = utils.read_json(path_entity_dict)
    entity_dict = {e["entity_id"]: e for e in entity_dict}
    print(f"Loaded entity dictionary with {len(entity_dict)} pages")

    target_tree_numbers = load_target_tree_numbers(
        path="../dataset-meta-information/umls_interesting_semantic_types.csv"
    )

    print("Filtering triples ...")
    new_triples = []
    for triple in tqdm(triples):
        head_id = triple["head"]
        tail_id = triple["tail"]
        head_tree_numbers = entity_dict[head_id]["tree_numbers"]
        tail_tree_numbers = entity_dict[tail_id]["tree_numbers"]
        head_ok = False
        tail_ok = False
        for head_num in head_tree_numbers:
            if a_is_under_bs(head_num, target_tree_numbers):
                head_ok = True
                break
        for tail_num in tail_tree_numbers:
            if a_is_under_bs(tail_num, target_tree_numbers):
                tail_ok = True
                break
        if head_ok and tail_ok:
            new_triples.append(triple)
    print(f"The number of triples is reduced from {len(triples)} to {len(new_triples)}")
    
    utils.write_json(path_output_triples, new_triples)
    print(f"Saved the resulting triples to {path_output_triples}")

    print("Done.")

    
def load_target_tree_numbers(path):
    df = pd.read_csv(path)
    target_tree_numbers = df.set_index("Type")["Tree Number"].to_dict()
    target_tree_numbers = list(target_tree_numbers.values())
    return target_tree_numbers

    
def a_is_under_bs(a, bs):
    for b in bs:
        if b in a:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_triples", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, required=True)
    parser.add_argument("--output_triples", type=str, required=True)
    args = parser.parse_args()
    main(args)