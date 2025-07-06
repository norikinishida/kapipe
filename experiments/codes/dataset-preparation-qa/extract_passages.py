import argparse
import json
import os

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_input_file = args.input_file
    path_output_file = args.output_file

    utils.mkdir(os.path.dirname(path_output_file))

    contexts = utils.read_json(path_input_file)
    print(f"Loaded gold contexts for {len(contexts)} questions from {path_input_file}")

    passages = [tuple(c.items()) for contexts_for_question in contexts for c in contexts_for_question["contexts"]]
    passages = list(set(passages)) # list[tuple[tuple]]
    print(f"Extracted {len(passages)} unique passages")
    passages = [
        dict(p_tuples)
        for p_tuples in passages
    ] # list[dict]
    passages = sorted(passages, key=lambda c: (c["url"], c["begin_section"], c["offset_in_begin_section"], c["end_section"], c["offset_in_end_section"]))

    with open(path_output_file, "w") as f:
        for passage in passages:
            json_str = json.dumps(passage)
            f.write(json_str + "\n")
    print(f"Saved {len(passages)} to {path_output_file}")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
