import argparse
import ast
import json
import os

from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_output_file = args.output_file

    utils.mkdir(os.path.dirname(path_output_file))

    dataset = load_dataset("ncbi/pubmed", trust_remote_code=True)
    print(f"Found {len(dataset['train'])} abstracts")
    
    with open(path_output_file, "w") as f:
        for data in tqdm(dataset["train"]):
            if data["MedlineCitation"]["Article"]["Language"] != "eng":
                continue
            doc_key = str(data["MedlineCitation"]["PMID"])
            text = data["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] 
            title = data["MedlineCitation"]["Article"]["ArticleTitle"]
            authors = data["MedlineCitation"]["Article"]["AuthorList"]["Author"]
            n_references = data["MedlineCitation"]["NumberOfReferences"]
            chemicals = data["MedlineCitation"]["ChemicalList"]["Chemical"]
            mesh_headings = data["MedlineCitation"]["MeshHeadingList"]["MeshHeading"]
            if doc_key.strip() and text.strip() and title.strip():
                passage = {
                    "title": title,
                    "text": text,
                    "pmid": pmid,
                    "authors": authors,
                    "n_references": n_references,
                    "chemicals": chemicals,
                    "mesh_headings": mesh_headings
                }
                json_str = json.dumps(passage)
                f.write(json_str + "\n")

    print("Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)

