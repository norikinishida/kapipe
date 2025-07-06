import argparse
import json
import os


import sys
sys.path.insert(0, "../../..")
from kapipe import utils
from kapipe.chunking import Chunker


def main(args):
    path_input_dir = args.input_dir
    path_output_file = args.output_file

    utils.mkdir(os.path.dirname(path_output_file))

    chunker = Chunker()

    count =0
    with open(path_output_file, "w") as f:
        # for split in ["train", "dev", "test"]:
        for split in ["train", "dev", "test"]:
            path_input_file = os.path.join(path_input_dir, split + ".json")
            documents = utils.read_json(path_input_file)
            print(f"Read {len(documents)} documents from {path_input_file}")
            for doc in documents:
                sentences = chunker.split_text_to_sentences(
                    text=" ".join(doc["sentences"])
                )
                assert len(sentences) > 1
                title = sentences[0]
                text = " ".join(sentences[1:])
                passage = {
                    "title": title,
                    "text": text,
                    "source_document": f"cdr/{split}/{doc['doc_key']}"
                }
                json_str = json.dumps(passage)
                f.write(json_str + "\n")
                count += 1
    
    print(f"Saved {count} passages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)

