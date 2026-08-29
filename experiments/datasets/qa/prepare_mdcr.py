import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

from tqdm import tqdm

from kapipe import utils


QUESTION_TEMPLATES: list[str] = [
    "Can I receive at least one of the following scholarship(s):",
    "Can I receive all the following scholarship(s):",
    (
        "What is the maximum number of scholarship(s) I can receive out of "
        "the following scholarship(s):"
    ),
]

QUESTION_TYPES: list[str] = [
    "any",
    "all",
    "maximum",
]


def main(args: argparse.Namespace) -> None:
    input_repository_dir: str = args.input_repository_dir
    output_articles_file: str = args.output_articles_file
    output_questions_file: str = args.output_questions_file
    output_gold_contexts_file: str = args.output_gold_contexts_file

    # Resolve every official MDCR input used by the conversion
    input_data_dir = os.path.join(
        input_repository_dir,
        "data",
        "scholarships",
    )
    input_docs_file = os.path.join(input_data_dir, "docs.json")
    input_parsed_file = os.path.join(input_data_dir, "parsed.json")
    input_questions_file = os.path.join(input_data_dir, "qs.json")
    input_relations_file = os.path.join(input_data_dir, "rels.json")
    input_gold_answer_file = os.path.join(
        input_repository_dir,
        "get_gold_ans.py",
    )

    # Fail before processing when the pinned repository is incomplete
    for input_file in [
        input_docs_file,
        input_parsed_file,
        input_questions_file,
        input_relations_file,
        input_gold_answer_file,
    ]:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Missing an MDCR input file: {input_file}")

    # Load and validate the document and atomic-condition annotations
    original_docs = utils.read_json(input_docs_file)
    parsed_docs = utils.read_json(input_parsed_file)
    assert isinstance(original_docs, list)
    assert isinstance(parsed_docs, list)
    assert len(original_docs) == len(parsed_docs)

    # Convert every HTML document to one newline-delimited corpus string
    articles, document_lines = convert_articles(original_docs=original_docs)

    # Load the official symbolic solver without changing its implementation
    get_gold_answer = load_official_gold_answer_function(
        input_repository_dir=input_repository_dir,
    )

    # Convert every scenario into the three official MDCR question types
    original_questions = utils.read_json(input_questions_file)
    assert isinstance(original_questions, list)
    questions = convert_questions(
        original_questions=original_questions,
        original_docs=original_docs,
        parsed_docs=parsed_docs,
        document_lines=document_lines,
        input_repository_dir=input_repository_dir,
        get_gold_answer=get_gold_answer,
    )

    # Materialize complete gold documents while keeping them out of questions
    document_id_to_document = {
        article["document_id"]: article
        for article in articles
    }
    gold_contexts = build_gold_contexts(
        questions=questions,
        document_id_to_document=document_id_to_document,
    )

    # Remove intermediate gold IDs from the model-facing question records
    for question in questions:
        del question["gold_document_ids"]

    # Create every output directory before writing converted records
    for output_file in [
        output_articles_file,
        output_questions_file,
        output_gold_contexts_file,
    ]:
        utils.mkdir(os.path.dirname(output_file))

    utils.write_jsonl(
        output_articles_file,
        articles,
        ensure_ascii=False,
    )
    utils.write_json(
        output_questions_file,
        questions,
        ensure_ascii=False,
    )
    utils.write_json(
        output_gold_contexts_file,
        gold_contexts,
        ensure_ascii=False,
    )

    print(
        f"Processed and saved {len(articles)} articles into "
        f"{output_articles_file}"
    )
    print(
        f"Processed and saved {len(questions)} questions into "
        f"{output_questions_file}"
    )
    print(
        f"Processed and saved {len(gold_contexts)} gold-context instances "
        f"into {output_gold_contexts_file}"
    )


def convert_articles(
    original_docs: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[list[str]]]:
    articles: list[dict[str, str]] = []
    document_lines: list[list[str]] = []

    # Preserve document order because MDCR uses list indices as document IDs
    for document_index, original_doc in enumerate(original_docs):
        assert isinstance(original_doc, dict)
        assert isinstance(original_doc["title"], str)
        assert isinstance(original_doc["url"], str)
        assert isinstance(original_doc["contents"], list)
        assert all(
            isinstance(content, str)
            for content in original_doc["contents"]
        )

        # Replace embedded newlines so one original content item remains one line
        lines = [
            remove_embedded_newlines(content=content)
            for content in original_doc["contents"]
        ]
        assert all("\n" not in line and "\r" not in line for line in lines)

        text = "\n".join(lines)
        assert text.split("\n") == lines

        article = {
            "title": original_doc["title"],
            "text": text,
            "document_id": str(document_index),
            "url": original_doc["url"],
        }
        articles.append(article)
        document_lines.append(lines)

    return articles, document_lines


def remove_embedded_newlines(content: str) -> str:
    # Replace every common newline encoding without otherwise normalizing HTML
    return content.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def load_official_gold_answer_function(
    input_repository_dir: str,
) -> Callable[[str, list[int], int, list[str], list[bool]], dict[str, Any]]:
    # Provide the single tiger_utils function imported by the official code
    tiger_utils_module = ModuleType("tiger_utils")

    def read_json(path: str) -> Any:
        # Read official JSON files relative to the MDCR repository
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    tiger_utils_module.read_json = read_json  # type: ignore[attr-defined]
    sys.modules["tiger_utils"] = tiger_utils_module

    # Import the pinned official modules before restoring the search path
    sys.path.insert(0, input_repository_dir)
    try:
        gold_answer_module = importlib.import_module("get_gold_ans")
    finally:
        del sys.path[0]

    module_file = os.path.abspath(gold_answer_module.__file__)
    expected_module_file = os.path.abspath(
        os.path.join(input_repository_dir, "get_gold_ans.py")
    )
    if module_file != expected_module_file:
        raise ImportError(
            "Loaded get_gold_ans.py from an unexpected location: "
            f"{module_file}"
        )

    get_gold_answer = gold_answer_module.get_gold_ans
    assert callable(get_gold_answer)
    return get_gold_answer


def convert_questions(
    original_questions: list[dict[str, Any]],
    original_docs: list[dict[str, Any]],
    parsed_docs: list[dict[str, Any]],
    document_lines: list[list[str]],
    input_repository_dir: str,
    get_gold_answer: Callable[
        [str, list[int], int, list[str], list[bool]],
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    # Run the official solver from its repository-relative data directory
    original_working_dir = os.getcwd()
    os.chdir(input_repository_dir)
    try:
        for scenario_index, original_question in enumerate(
            tqdm(original_questions, desc="Converting MDCR questions")
        ):
            # Validate every scenario field consumed by the official solver
            assert isinstance(original_question, dict)
            assert isinstance(original_question["doc_idxs"], list)
            assert all(
                isinstance(document_index, int)
                for document_index in original_question["doc_idxs"]
            )
            assert isinstance(original_question["given_conditions"], list)
            assert all(
                isinstance(condition_id, str)
                for condition_id in original_question["given_conditions"]
            )
            assert isinstance(original_question["given_values"], list)
            assert all(
                isinstance(value, bool)
                for value in original_question["given_values"]
            )
            assert len(original_question["given_conditions"]) == len(
                original_question["given_values"]
            )
            assert isinstance(original_question["scenario"], str)

            document_indices = original_question["doc_idxs"]
            document_titles = [
                original_docs[document_index]["title"]
                for document_index in document_indices
            ]
            gold_document_ids = [
                str(document_index)
                for document_index in document_indices
            ]

            # Create the any, all, and maximum questions for this scenario
            for question_type_index, question_template in enumerate(
                QUESTION_TEMPLATES
            ):
                gold_answer = get_gold_answer(
                    "scholarships",
                    document_indices,
                    question_type_index,
                    original_question["given_conditions"],
                    original_question["given_values"],
                )
                assert isinstance(gold_answer, dict)
                assert isinstance(gold_answer["answer"], (str, int))

                # Preserve every official missing-condition group and atom
                original_condition_groups = gold_answer.get("conditions", [])
                assert isinstance(original_condition_groups, list)
                conditions = convert_condition_groups(
                    original_condition_groups=original_condition_groups,
                    parsed_docs=parsed_docs,
                    document_lines=document_lines,
                )
                assert len(conditions) == len(original_condition_groups)
                assert all(
                    len(condition_group) == len(original_condition_group)
                    for condition_group, original_condition_group in zip(
                        conditions,
                        original_condition_groups,
                        strict=True,
                    )
                )

                question_key = (
                    f"mdcr-scholarships-{scenario_index:04d}-"
                    f"{QUESTION_TYPES[question_type_index]}"
                )
                question_text = "\n".join(
                    [question_template, *document_titles]
                )

                question = {
                    "question_key": question_key,
                    "question": question_text,
                    "scenario": original_question["scenario"],
                    "answers": [
                        {
                            "answer": str(gold_answer["answer"]),
                        }
                    ],
                    "conditions": conditions,
                    "question_type": QUESTION_TYPES[question_type_index],
                    "gold_document_ids": gold_document_ids,
                }
                questions.append(question)
    finally:
        os.chdir(original_working_dir)

    return questions


def convert_condition_groups(
    original_condition_groups: list[list[str]],
    parsed_docs: list[dict[str, Any]],
    document_lines: list[list[str]],
) -> list[list[dict[str, Any]]]:
    condition_groups: list[list[dict[str, Any]]] = []

    # Preserve outer OR groups and inner AND atoms in their original order
    for original_condition_group in original_condition_groups:
        assert isinstance(original_condition_group, list)
        assert all(
            isinstance(condition_id, str)
            for condition_id in original_condition_group
        )

        condition_group = [
            convert_condition(
                condition_id=condition_id,
                parsed_docs=parsed_docs,
                document_lines=document_lines,
            )
            for condition_id in original_condition_group
        ]
        condition_groups.append(condition_group)

    return condition_groups


def convert_condition(
    condition_id: str,
    parsed_docs: list[dict[str, Any]],
    document_lines: list[list[str]],
) -> dict[str, Any]:
    # Parse the official doc{int}-c{int} atomic-condition identifier
    document_id_text, condition_name = condition_id.split("-", 1)
    assert document_id_text.startswith("doc")
    assert condition_name.startswith("c")
    document_index = int(document_id_text.removeprefix("doc"))

    parsed_doc = parsed_docs[document_index]
    assert isinstance(parsed_doc, dict)
    assert isinstance(parsed_doc["conditions"], dict)
    condition_value = parsed_doc["conditions"][condition_name]

    # Resolve direct source references without changing their multiplicity
    if isinstance(condition_value, int):
        line_numbers = [condition_value]
        descriptions = [document_lines[document_index][condition_value]]
    elif isinstance(condition_value, list):
        assert all(isinstance(line_number, int) for line_number in condition_value)
        line_numbers = list(condition_value)
        descriptions = [
            document_lines[document_index][line_number]
            for line_number in line_numbers
        ]
    else:
        # Preserve split atomic-condition descriptions exactly as annotated
        assert isinstance(condition_value, str)
        assert isinstance(parsed_doc["mapping"], dict)
        mapping_value = parsed_doc["mapping"][condition_name]
        if isinstance(mapping_value, int):
            line_numbers = [mapping_value]
        else:
            assert isinstance(mapping_value, list)
            assert all(
                isinstance(line_number, int)
                for line_number in mapping_value
            )
            line_numbers = list(mapping_value)
        descriptions = [condition_value]

    # Validate every source location against the newline-delimited corpus
    assert line_numbers
    assert all(
        0 <= line_number < len(document_lines[document_index])
        for line_number in line_numbers
    )
    assert descriptions
    assert all(isinstance(description, str) for description in descriptions)

    return {
        "document_id": str(document_index),
        "line_numbers": line_numbers,
        "descriptions": descriptions,
    }


def build_gold_contexts(
    questions: list[dict[str, Any]],
    document_id_to_document: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gold_contexts: list[dict[str, Any]] = []

    # Resolve every hidden MDCR document index to its complete corpus record
    for question in questions:
        contexts = [
            document_id_to_document[document_id]
            for document_id in question["gold_document_ids"]
        ]
        gold_contexts.append(
            {
                "question_key": question["question_key"],
                "contexts": contexts,
            }
        )

    return gold_contexts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_repository_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_articles_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_questions_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_gold_contexts_file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
