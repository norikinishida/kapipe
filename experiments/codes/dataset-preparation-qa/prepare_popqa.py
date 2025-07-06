import argparse
import ast
import os

from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_output_dir = args.output_dir
    size = args.size

    utils.mkdir(path_output_dir)

    dataset = load_dataset("akariasai/PopQA")

    process_split(
        split_dataset=dataset["test"],
        path_output_file=os.path.join(path_output_dir, "test.json")
    )
    if size > 0:
        process_split(
            split_dataset=dataset["test"],
            path_output_file=os.path.join(path_output_dir, f"test_{size}.json"),
            size=size
        )


def process_split(split_dataset, path_output_file, size=None):
    questions = []
    for data in tqdm(split_dataset):
        # Extract example ID
        question_key = data["id"]

        # Extract question
        question = data["question"] 

        # Extract contexts
        # contexts = []

        # Extract triple
        head = {
            "name": data["subj"],
            "aliases": ast.literal_eval(data["s_aliases"]),
            "wikidata_id": data["subj_id"], 
            "wikidata_uri": data["s_uri"],
            "wikipedia_title": data["s_wiki_title"],
            "wikipedia_monthly_pageview": data["s_pop"]
        }
        tail = {
            "name": data["obj"],
            "aliases": ast.literal_eval(data["o_aliases"]),
            "wikidata_id": data["obj_id"],
            "wikidata_uri": data["o_uri"],
            "wikipedia_title": data["o_wiki_title"],
            "wikipedia_monthly_pageview": data["o_pop"]
        }
        relation = {
            "name": data["prop"],
            "wikidata_id": data["prop_id"]
        }
        triple = {
            "head": head,
            "tail": tail,
            "relation": relation
        }
        triples = [triple]

        # Aggregate answers
        answers = []
        possible_answers = data["possible_answers"]
        assert isinstance(possible_answers, str)
        possible_answers = ast.literal_eval(possible_answers)
        assert isinstance(possible_answers, list)
        for answer_text in possible_answers:
            answers.append({
                "answer": answer_text
            })            

        if len([a for a in answers if a["answer"].strip() != ""]) == 0:
            continue

        question = {
            "question_key": question_key,
            "question": question,
            "answers": answers,
            # "contexts": contexts,
            "triples": triples,
        }
        documents.append(document)
        if size is not None and len(documents) >= size:
            break
        
    utils.write_json(path_output_file, documents)
    print(f"Processed and saved {len(documents)} questions into {path_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--size", type=int, default=-1)
    args = parser.parse_args()
    main(args)

 