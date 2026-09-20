import argparse
import ast
import csv
import os

from tqdm import tqdm

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    input_file_path: str = args.input_file
    output_file_path: str = args.output_file
    size: int = args.size

    # Create the output directory when necessary
    output_dir = os.path.dirname(output_file_path)
    if output_dir != "":
        utils.mkdir(output_dir)

    # Build stable question keys from the DPR file name
    question_key_base = os.path.basename(input_file_path)
    assert question_key_base.endswith(".qa.csv")
    question_key_base = question_key_base.removesuffix(".qa.csv")
    dataset_name, split = question_key_base.rsplit("-", maxsplit=1)
    if dataset_name == "trivia":
        dataset_name = "triviaqa"

    questions: list[dict[str, object]] = []

    # Read the DPR tab-separated QA file
    with open(
        input_file_path,
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.reader(f, delimiter="\t")

        for row_i, row in tqdm(enumerate(reader)):
            # Fail when the input does not follow the DPR format.
            assert len(row) == 2, row

            question_text = row[0].replace("’", "'")

            # Parse the serialized list of acceptable answer aliases.
            answer_texts = ast.literal_eval(row[1])
            assert isinstance(answer_texts, list)
            assert len(answer_texts) > 0
            assert all(isinstance(answer_text, str) for answer_text in answer_texts)

            answers = [
                {
                    "answer": answer_text,
                }
                for answer_text in answer_texts
            ]

            question = {
                "question_key": f"{dataset_name}/{split}/{row_i}",
                "question": question_text,
                "answers": answers,
            }
            questions.append(question)

            # Stop after producing the requested subset.
            if size > 0 and len(questions) >= size:
                break

    utils.write_json(output_file_path, questions)

    print(
        f"Processed and saved {len(questions)} questions "
        f"into {output_file_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=-1,
    )

    args = parser.parse_args()
    main(args)
