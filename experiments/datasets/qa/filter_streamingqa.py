import argparse
import os
from typing import Any

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    input_questions_dir: str = args.input_questions_dir
    output_questions_dir: str = args.output_questions_dir

    # Create the output directory before processing all dataset splits
    utils.mkdir(output_questions_dir)

    # Filter every prepared split without changing question order
    for split in ["train", "dev", "test"]:
        input_questions_file = os.path.join(
            input_questions_dir,
            f"{split}.json",
        )
        output_questions_file = os.path.join(
            output_questions_dir,
            f"{split}_filtered.json",
        )

        # Load the complete prepared question split
        questions = utils.read_json(input_questions_file)

        # Remove questions unsuitable for single-answer QA
        filtered_questions = filter_questions(questions=questions)

        # Save the filtered questions in their original order
        utils.write_json(
            output_questions_file,
            filtered_questions,
            ensure_ascii=False,
        )
        print(
            f"Processed and saved {len(filtered_questions)} filtered "
            f"questions into {output_questions_file}"
        )


def filter_questions(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filtered_questions: list[dict[str, Any]] = []

    # Remove yes/no questions and questions unsuitable for single-answer QA
    for question in questions:
        # Remove yes/no questions
        if any(
            answer["answer"].strip().lower() in {"yes", "no"}
            for answer in question["answers"]
        ):
            continue

        # Remove questions with multiple correct answers
        if len(question["answers"]) != 1:
            continue

        # Copy the retained question without adding MCQA placeholder options
        filtered_questions.append(dict(question))

    return filtered_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_questions_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_questions_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
