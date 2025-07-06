import argparse
import json
import os


import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_input_passages = args.input_passages
    path_input_gold_contexts = args.input_gold_contexts
    path_output_passages = args.output_passages

    utils.mkdir(os.path.dirname(path_output_passages))

    print(f"Loading gold contexts from {path_input_gold_contexts} ...")
    gold_contexts = utils.read_json(path_input_gold_contexts)
    print(f"Loaded {len(gold_contexts)} gold contexts")
    possible_pubmed_ids = set()
    prefix = "http://www.ncbi.nlm.nih.gov/pubmed/"
    for contexts_for_q in gold_contexts:
        for context in contexts_for_q["contexts"]:
            assert prefix in context["url"]
            pmid = context["url"].replace(prefix, "")
            possible_pubmed_ids.add(pmid)
    print(f"Found {len(possible_pubmed_ids)} unique PubMed IDs in total")

    print(f"Loading PubMed abstracts from {path_input_passages} ...")
    passages = []
    with open(path_input_passages) as f:
        for line in f:
            json_obj = json.loads(line.strip())
            if json_obj["pmid"] in possible_pubmed_ids:
                passages.append(json_obj)

    print(f"Saving the results to {path_output_passages}") 
    with open(path_output_passages, "w") as f:
        for passage in passages:
            json_str = json.dumps(passage)
            f.write(json_str + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_passages", type=str, required=True)
    parser.add_argument("--input_gold_contexts", type=str, required=True)
    parser.add_argument("--output_passages", type=str, required=True)
    args = parser.parse_args()
    main(args)

