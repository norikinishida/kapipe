import argparse
import os

from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe.demonstration_retrieval import DemonstrationRetriever
from kapipe import utils


def main(args):
    retriever = DemonstrationRetriever(
        path_demonstration_pool=args.demonstration_pool,
        method=args.method,
        task=args.task,
        path_entity_dict=args.entity_dict
    )

    documents = utils.read_json(args.documents)

    demonstrations = []
    for document in tqdm(documents):
        demonstrations_for_doc = retriever.search(
            document=document,
            top_k=args.n_demos
        )
        demonstrations.append(demonstrations_for_doc)

    utils.mkdir(os.path.dirname(args.output_file))
    utils.write_json(args.output_file, demonstrations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=str, required=True)
    parser.add_argument("--n_demos", type=int, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--demonstration_pool", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
