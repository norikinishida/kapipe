import argparse
import os
import random
from typing import Any

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    input_questions_file: str = args.input_questions_file
    output_questions_file: str = args.output_questions_file
    random_seed: int = args.random_seed

    # Load the complete CLARK-News question collection
    questions = utils.read_json(input_questions_file)

    # Select temporal multiple-choice questions
    filtered_questions = filter_questions(
        questions=questions,
        random_seed=random_seed,
    )

    # Create the output directory before writing the filtered collection
    output_dir = os.path.dirname(output_questions_file)
    if output_dir:
        utils.mkdir(output_dir)
    utils.write_json(
        output_questions_file,
        filtered_questions,
        ensure_ascii=False,
    )

    print(
        f"Processed and saved {len(filtered_questions)} filtered questions "
        f"into {output_questions_file}"
    )


def filter_questions(
    questions: list[dict[str, Any]],
    random_seed: int,
) -> list[dict[str, Any]]:
    # Remove question types that do not support temporal multiple-choice QA
    eligible_questions: list[dict[str, Any]] = []
    for question in questions:
        # Remove yes/no questions
        if any(
            answer["answer"].lower() in {"yes", "no"}
            for answer in question["answers"]
        ):
            continue

        # Remove questions whose supporting fact could not be identified
        if None in question["triples"]:
            continue

        # Remove questions with multiple correct answers at the same timestamp
        if len(question["answers"]) != 1:
            continue

        eligible_questions.append(question)

    # Group the remaining questions by their time-agnostic question text
    question_text_to_questions: dict[str, list[dict[str, Any]]] = {}
    for question in eligible_questions:
        question_text_to_questions.setdefault(
            question["question"],
            [],
        ).append(question)

    # Retain question groups with at least three temporal observations
    question_groups = [
        group
        for group in question_text_to_questions.values()
        if len(group) >= 3
    ]
    question_groups.sort(key=len, reverse=True)

    # Use the answers at other timestamps as temporal distractors
    rng = random.Random(random_seed)
    filtered_questions: list[dict[str, Any]] = []
    for group in question_groups:
        candidate_answers = [
            question["answers"][0]["answer"]
            for question in group
        ]
        rng.shuffle(candidate_answers)

        for question in group:
            answer_text = question["answers"][0]["answer"]
            filtered_question = dict(question)
            filtered_question["answers"] = [
                {
                    "answer": answer_text,
                }
            ]
            filtered_question["candidate_answers"] = list(candidate_answers)
            filtered_questions.append(filtered_question)

    return filtered_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_questions_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_questions_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    main(args)
