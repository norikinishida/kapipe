import argparse
import json
import os
from typing import Any

from tqdm import tqdm

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    input_dev_file: str = args.input_dev_file
    input_test_file: str = args.input_test_file
    input_articles_file: str = args.input_articles_file
    output_dir: str = args.output_dir

    # Validate every input before processing the large Wikipedia corpus
    for input_file in [
        input_dev_file,
        input_test_file,
        input_articles_file,
    ]:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Missing an input file: {input_file}")

    # Convert the official question annotations and collect their evidence
    split_to_questions: dict[str, list[dict[str, Any]]] = {}
    split_to_evidence_lists: dict[str, list[list[dict[str, Any]]]] = {}
    required_titles: set[str] = set()

    for split, input_file in [
        ("dev", input_dev_file),
        ("test", input_test_file),
    ]:
        questions, evidence_lists = load_questions(
            input_file=input_file,
            split=split,
        )
        split_to_questions[split] = questions
        split_to_evidence_lists[split] = evidence_lists

        # Collect every unique Wikipedia title needed by either split
        for evidence_list in evidence_lists:
            for evidence in evidence_list:
                required_titles.add(evidence["title"])

    # Read the full Wikipedia corpus once and retain only annotated evidence
    title_to_article = load_evidence_articles(
        input_articles_file=input_articles_file,
        required_titles=required_titles,
    )

    # Create the output directory before writing both official splits
    utils.mkdir(output_dir)

    for split in ["dev", "test"]:
        questions = split_to_questions[split]
        evidence_lists = split_to_evidence_lists[split]
        incomplete_question_count = 0

        # Annotate every question without removing incomplete instances
        for question, evidence_list in zip(
            questions,
            evidence_lists,
            strict=True,
        ):
            missing_titles = sorted(
                evidence["title"]
                for evidence in evidence_list
                if evidence["title"] not in title_to_article
            )

            if split == "dev":
                # Convert every recursive step to the decomposition format
                question["question_decomposition"] = (
                    convert_question_decomposition(
                        decomposition=question["question_decomposition"],
                        title_to_article=title_to_article,
                    )
                )

            question["full_found_in_enwiki2023"] = not missing_titles

            if missing_titles:
                incomplete_question_count += 1
                print(
                    f"Incomplete {split} question "
                    f"{question['question_key']}: missing articles "
                    f"{missing_titles}"
                )

        # Report the number of questions with unavailable evidence
        print(
            f"Found incomplete evidence for {incomplete_question_count} of "
            f"{len(questions)} {split} questions"
        )

        gold_contexts = build_gold_contexts(
            questions=questions,
            evidence_lists=evidence_lists,
            title_to_article=title_to_article,
        )

        output_questions_file = os.path.join(output_dir, f"{split}.json")
        output_gold_contexts_file = os.path.join(
            output_dir,
            f"{split}.gold_contexts.json",
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
            f"Processed and saved {len(questions)} questions into "
            f"{output_questions_file}"
        )
        print(
            f"Processed and saved {len(gold_contexts)} gold-context "
            f"instances into {output_gold_contexts_file}"
        )


def load_questions(
    input_file: str,
    split: str,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    # Load the official list because each release file contains one split
    original_questions = utils.read_json(input_file)
    assert isinstance(original_questions, list)

    questions: list[dict[str, Any]] = []
    evidence_lists: list[list[dict[str, Any]]] = []

    for data in tqdm(original_questions, desc=f"Processing FanOutQA {split}"):
        # Validate the fields shared by the development and test splits
        assert isinstance(data, dict)
        assert isinstance(data["id"], str)
        assert isinstance(data["question"], str)

        question_key = f"fanoutqa-{split}-{data['id']}"
        question: dict[str, Any] = {
            "question_key": question_key,
            "question": data["question"],
        }

        if split == "dev":
            # Preserve the recursive decomposition as dataset-specific metadata
            decomposition = data["decomposition"]
            assert isinstance(decomposition, list)
            question["answers"] = [
                {
                    "answer": convert_answer_to_text(data["answer"]),
                }
            ]
            question["question_decomposition"] = decomposition

            # Derive the necessary evidence from every recursive node
            evidence_list = collect_decomposition_evidence(
                decomposition=decomposition,
            )
        else:
            assert split == "test"

            # Keep hidden test answers and decompositions absent
            original_evidence_list = data["necessary_evidence"]
            assert isinstance(original_evidence_list, list)
            evidence_list = deduplicate_evidence(
                evidence_list=original_evidence_list,
            )

        questions.append(question)
        evidence_lists.append(evidence_list)

    return questions, evidence_lists


def convert_answer_to_text(answer: Any) -> str:
    # Preserve ordinary string answers without adding JSON quotation marks
    if isinstance(answer, str):
        return answer

    # Serialize structured FanOutQA answers deterministically and losslessly
    assert isinstance(answer, (bool, int, float, list, dict))
    return json.dumps(
        answer,
        ensure_ascii=False,
        sort_keys=True,
    )


def collect_decomposition_evidence(
    decomposition: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_list: list[dict[str, Any]] = []

    # Traverse the official recursive decomposition in its original order
    for step in decomposition:
        assert isinstance(step, dict)
        assert isinstance(step["id"], str)
        assert isinstance(step["question"], str)
        assert isinstance(step["decomposition"], list)
        assert isinstance(step["depends_on"], list)
        assert all(
            isinstance(dependency_id, str)
            for dependency_id in step["depends_on"]
        )

        evidence = step["evidence"]
        if evidence is not None:
            evidence_list.append(evidence)

        evidence_list.extend(
            collect_decomposition_evidence(
                decomposition=step["decomposition"],
            )
        )

    return deduplicate_evidence(evidence_list=evidence_list)


def deduplicate_evidence(
    evidence_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduplicated_evidence: list[dict[str, Any]] = []
    title_to_evidence: dict[str, dict[str, Any]] = {}

    # Preserve the first occurrence while validating repeated annotations
    for evidence in evidence_list:
        assert isinstance(evidence, dict)
        assert isinstance(evidence["pageid"], int)
        assert isinstance(evidence["revid"], int)
        assert isinstance(evidence["title"], str)
        assert isinstance(evidence["url"], str)

        # Remove section fragments because gold contexts contain complete pages
        normalized_evidence = dict(evidence)
        normalized_evidence["url"] = evidence["url"].split("#", 1)[0]

        title = normalized_evidence["title"]
        if title in title_to_evidence:
            continue

        # Keep the first annotation because a title identifies one document
        title_to_evidence[title] = normalized_evidence
        deduplicated_evidence.append(normalized_evidence)

    return deduplicated_evidence


def load_evidence_articles(
    input_articles_file: str,
    required_titles: set[str],
) -> dict[str, dict[str, Any]]:
    title_to_article: dict[str, dict[str, Any]] = {}

    # Stream the large corpus instead of loading every Wikipedia article
    with open(input_articles_file, encoding="utf-8") as file:
        for line in tqdm(file, desc="Finding FanOutQA evidence articles"):
            article = json.loads(line)
            title = article["title"]
            if title not in required_titles:
                continue

            assert title not in title_to_article
            assert isinstance(title, str)
            assert isinstance(article["id"], str)
            assert isinstance(article["revid"], str)
            assert isinstance(article["url"], str)
            assert isinstance(article["text"], str)
            assert article["text"].strip()
            title_to_article[title] = article

    return title_to_article


def convert_question_decomposition(
    decomposition: list[dict[str, Any]],
    title_to_article: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    converted_decomposition: list[dict[str, Any]] = []

    # Convert every step while preserving the official recursive structure
    for step in decomposition:
        evidence = step["evidence"]
        contexts = [] if evidence is None else [
            build_decomposition_context(
                evidence=evidence,
                title_to_article=title_to_article,
            )
        ]

        # Place the decomposition fields in the repository-standard order
        converted_step: dict[str, Any] = {
            "id": step["id"],
            "depends_on": step["depends_on"],
            "question": step["question"],
            "answers": convert_decomposition_answers(step["answer"]),
            "contexts": contexts,
        }

        # Preserve nested decompositions after the common QA fields
        if step["decomposition"]:
            converted_step["question_decomposition"] = (
                convert_question_decomposition(
                    decomposition=step["decomposition"],
                    title_to_article=title_to_article,
                )
            )

        converted_decomposition.append(converted_step)

    return converted_decomposition


def convert_decomposition_answers(answer: Any) -> list[dict[str, Any]]:
    # Preserve every list element as an independently evaluable answer
    if isinstance(answer, list):
        return [
            {
                "answer": convert_answer_to_text(answer_element),
                "answer_type": "list",
                "list_index": list_index,
            }
            for list_index, answer_element in enumerate(answer)
        ]

    # Preserve a scalar or structured non-list answer as one text answer
    return [
        {
            "answer": convert_answer_to_text(answer),
        }
    ]


def build_decomposition_context(
    evidence: dict[str, Any],
    title_to_article: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    title = evidence["title"]

    # Use corpus metadata when the title exists in the snapshot
    if title in title_to_article:
        article = title_to_article[title]
        return {
            "pageid": int(article["id"]),
            "revid": int(article["revid"]),
            "url": article["url"],
            "title": article["title"],
            "found_in_enwiki2023": True,
        }

    # Retain the evidence metadata when the corpus has no matching title
    return {
        "pageid": evidence["pageid"],
        "revid": evidence["revid"],
        "url": evidence["url"],
        "title": title,
        "found_in_enwiki2023": False,
    }


def build_context(
    evidence: dict[str, Any],
    title_to_article: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    # Preserve the corpus field order and append the availability flag
    if evidence["title"] in title_to_article:
        article = title_to_article[evidence["title"]]
        return {
            "pageid": int(article["id"]),
            "revid": int(article["revid"]),
            "url": article["url"],
            "title": article["title"],
            "text": article["text"],
            "found_in_enwiki2023": True,
        }

    # Retain unavailable evidence as an empty Passage with source metadata
    return {
        "pageid": evidence["pageid"],
        "revid": evidence["revid"],
        "url": evidence["url"],
        "title": evidence["title"],
        "text": "",
        "found_in_enwiki2023": False,
    }


def build_gold_contexts(
    questions: list[dict[str, Any]],
    evidence_lists: list[list[dict[str, Any]]],
    title_to_article: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    assert len(questions) == len(evidence_lists)
    gold_contexts: list[dict[str, Any]] = []

    for question, evidence_list in zip(
        questions,
        evidence_lists,
        strict=True,
    ):
        contexts: list[dict[str, Any]] = []

        # Preserve the official evidence order for each question
        for evidence in evidence_list:
            contexts.append(
                build_context(
                    evidence=evidence,
                    title_to_article=title_to_article,
                )
            )

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
        "--input_dev_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_test_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_articles_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
