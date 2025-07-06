import argparse
import os

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


# NOTE: This list should be made based on the coarse-grained relation labels
IGNORED_LABELS = [
    "is_a",
    "is_sibling_with",
    "is_equivalent_to",
    "is_classified_as",
    "has_variant",
    "misc."
]


def main(args):
    path_input_triples = args.input_triples
    path_output_triples = args.output_triples

    utils.mkdir(os.path.dirname(path_output_triples))

    print(f"Loading triples from {path_input_triples}")
    triples = utils.read_json(path_input_triples)
    print(f"Loaded {len(triples)} triples")

    relation_mapping = load_relation_mapping(
        path_relation_mapping="../dataset-meta-information/umls2017aa_rela_to_coarse_grained_relation_label.csv"
    )

    print("Mapping relation labels ...")
    new_triples = []
    memo = set()
    for triple in tqdm(triples):
        head_id = triple["head"]
        tail_id = triple["tail"]
        old_rel = triple["relation"]

        new_rel_info = relation_mapping[old_rel]
        new_rel_label = new_rel_info["label"]

        # Skip unimportant relations
        if new_rel_label in IGNORED_LABELS:
            continue

        # Skip negative relations
        if new_rel_info["is_negative"]:
            continue

        # Swap the head and tail if necessary
        if new_rel_info["is_inverse"]:
            tmp = tail_id
            tail_id = head_id
            head_id = tmp

        memo_item = (head_id, tail_id, new_rel_label)
        if not memo_item in memo:
            memo.add(memo_item)
            new_triples.append({
                "head": head_id,
                "tail": tail_id,
                "relation": new_rel_label
            })

    print(f"The number of triples is reduced from {len(triples)} to {len(new_triples)}")

    utils.write_json(path_output_triples, new_triples)
    print(f"Saved the resulting triples to {path_output_triples}")

    print("Done.")


def load_relation_mapping(path_relation_mapping):
    # Get a mapping from RELA to coarse-grained relation label
    df = pd.read_csv(path_relation_mapping)
    relation_mapping = df.set_index("RELA")["Coarse-Grained Relation Label"].to_dict()
    relation_mapping = {k: {"label": v} for k,v in relation_mapping.items()}

    # Set inverse and negative indicators
    for k, v in relation_mapping.items():
        # Set inverse indicator
        v["is_inverse"] = "#inverse" in v["label"]
        # Set negative indicator
        v["is_negative"] = "#negative" in v["label"]
        # Set label
        v["label"] = v["label"].replace("#inverse", "")
        v["label"] = v["label"].replace("#negative", "")

    return relation_mapping



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_triples", type=str, required=True)
    parser.add_argument("--output_triples", type=str, required=True)
    args = parser.parse_args()
    main(args)