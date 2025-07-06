import argparse
import ast
import csv
import os

from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_input_file = args.input_file
    from_json = args.from_json
    context_type = args.context_type
    path_output_file = args.output_file
    size = args.size

    utils.mkdir(os.path.dirname(path_output_file))

    questions = []
    contexts = []

    question_key_base = os.path.basename(path_input_file)

    if from_json:
        assert context_type is not None
        dataset = utils.read_json(path_input_file)
        for data_i, data in tqdm(enumerate(dataset), total=len(dataset)):
            question_key = f"{question_key_base}-{data_i}"
            question = data["question"]

            answers = data["answers"]
            assert isinstance(answers, list)
            answers = [{"answer": a} for a in answers]

            contexts_for_question = []
            for ctx in data["positive_ctxs"]:
                context = ctx["text"]
                title = ctx["title"]
                annotation_score = ctx["score"]
                if annotation_score == 1000:
                    annotation_type = "gold"
                else:
                    annotation_type = "distant"
                annotation_title_score = ctx["title_score"]
                if "passage_id" in ctx:
                    passage_id = ctx["passage_id"]
                elif "psg_id" in ctx:
                    passage_id = ctx["psg_id"]
                else:
                    raise ValueError(f"No passage-id key found: {ctx}")
                if context_type == annotation_type:
                    context = {
                        "title": title,
                        "text": context,
                        "passage_id": passage_id,
                        "annotation_type": annotation_type,
                        "annotation_score": annotation_score,
                        "annotation_title_score": annotation_title_score,
                    }
                    contexts_for_question.append(context)

            if len(contexts_for_question) == 0:
                assert context_type == "distant"
                continue

            question = {
                "question_key": question_key,
                "question": question,
                "answers": answers,
            }
            questions.append(question)

            contexts_for_question = {
                "question_key": question_key,
                "contexts": contexts_for_question,
            }
            contexts.append(contexts_for_question)

            if size > 0 and len(questions) >= size:
                break
    else:
        with open(path_input_file) as f:
            reader = csv.reader(f, delimiter="\t")
            for row_i, row in tqdm(enumerate(reader)):
                question_key = f"{question_key_base}-{row_i}"
                question = row[0]
                answers = ast.literal_eval(row[1])
                assert isinstance(answers, list)
                assert len(answers) > 0
                answers = [{"answer": a} for a in answers]
                question = {
                    "question_key": question_key,
                    "question": question,
                    "answers": answers,
                }
                questions.append(question)
                if size > 0 and len(questions) >= size:
                    break

    if from_json:
        assert len(questions) == len(contexts)
        path_output_file = path_output_file.replace(".json", f".{context_type}.json")
        utils.write_json(path_output_file, questions)
        print(f"Processed and saved {len(questions)} questions into {path_output_file}")
        utils.write_json(path_output_file.replace(".json", ".contexts.json"), contexts)
        print(f"Processed and saved {len(contexts)} contexts into {path_output_file.replace('.json', f'.contexts.json')}")
    else:
        utils.write_json(path_output_file, questions)
        print(f"Processed and saved {len(questions)} questions into {path_output_file}")
       

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--from_json", action="store_true")
    parser.add_argument("--context_type", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--size", type=int, default=-1)
    args = parser.parse_args()
    main(args)
