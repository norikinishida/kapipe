import argparse

from tqdm import tqdm

from kapipe import utils


def main(args):
    documents = utils.read_json(args.input_file)
    target_relations = args.target_relations
    print(f"Target Relations: {target_relations}")
    n_filtered = 0
    n_retained = 0
    for doc in tqdm(documents):
        triples = doc["relations"]
        n1 = len(triples)
        triples = [x for x in triples if x["relation"] in target_relations]
        n2 = len(triples)
        n_filtered += (n1 - n2)
        n_retained += n2
        doc["relations"] = triples
    print(f"Filtered {n_filtered} triples")
    print(f"{n_retained} triples are retained")
    utils.write_json(args.output_file, documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--target_relations", nargs="+", required=True)
    args = parser.parse_args()
    main(args=args)