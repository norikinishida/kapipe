import argparse
import json
import os
from typing import Any

from tqdm import tqdm

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    input_file: str = args.input_file
    output_articles_file: str = args.output_articles_file
    output_dir: str = args.output_dir

    # Validate that the input file exists before proceeding
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Missing a ConfRAG input file: {input_file}")

    questions: list[dict[str, Any]] = []
    gold_contexts: list[dict[str, Any]] = []
    seen_question_keys: set[str] = set()
    article_key_to_passage_key: dict[tuple[str, str], str] = {}

    # Convert each JSONL record independently because webpages are very long
    with open(input_file, encoding="utf-8") as file:
        for line_number, line in enumerate(
            tqdm(file, desc="Processing ConfRAG"),
            start=1,
        ):
            if not line.strip():
                continue

            data = json.loads(line)
            question, contexts_for_question = convert_instance(
                data=data,
                line_number=line_number,
                article_key_to_passage_key=article_key_to_passage_key,
            )

            # Validate that the question key is unique before adding it to the list
            question_key = question["question_key"]
            if question_key in seen_question_keys:
                raise ValueError(
                    f"Duplicate ConfRAG question key: {question_key}"
                )

            seen_question_keys.add(question_key)
            questions.append(question)
            gold_contexts.append(contexts_for_question)

    # Keep exactly one corpus article for each assigned passage key
    passage_key_to_article: dict[str, dict[str, Any]] = {}
    for contexts_for_question in gold_contexts:
        for context in contexts_for_question["contexts"]:
            passage_key = context["passage_key"]
            if passage_key in passage_key_to_article:
                existing_article = passage_key_to_article[passage_key]
                if (
                    context["website"] != existing_article["website"]
                    or context["text"] != existing_article["text"]
                ):
                    raise ValueError(
                        f"Conflicting ConfRAG article: {passage_key}"
                    )
            else:
                passage_key_to_article[passage_key] = context

    articles = list(passage_key_to_article.values())
    if len(articles) != len(article_key_to_passage_key):
        raise ValueError("Incomplete ConfRAG article mapping")

    # Create the destination directories only after all records are validated
    utils.mkdir(output_dir)
    utils.mkdir(os.path.dirname(output_articles_file))
    output_questions_file = os.path.join(output_dir, "train.json")
    output_gold_contexts_file = os.path.join(
        output_dir,
        "train.gold_contexts.json",
    )

    # Save the complete webpages as newline-delimited retrieval records
    utils.write_jsonl(
        output_articles_file,
        articles,
        ensure_ascii=False,
    )

    # Save questions separately from their complete gold contexts
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
        f"Processed and saved {len(articles)} ConfRAG articles into "
        f"{output_articles_file}"
    )
    print(
        f"Processed and saved {len(questions)} ConfRAG questions into "
        f"{output_questions_file}"
    )
    print(
        f"Processed and saved {len(gold_contexts)} gold-context instances "
        f"into {output_gold_contexts_file}"
    )


def convert_instance(
    data: dict[str, Any],
    line_number: int,
    article_key_to_passage_key: dict[tuple[str, str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Detect changes to the official recommended ConfRAG schema immediately
    assert set(data) == {
        "id",
        "question",
        "websites",
        "from",
        "contradicts",
        "answers",
    }, line_number
    assert isinstance(data["id"], int), line_number
    assert isinstance(data["question"], str), line_number
    assert isinstance(data["websites"], list), line_number
    assert isinstance(data["from"], str), line_number
    assert isinstance(data["contradicts"], bool), line_number
    assert isinstance(data["answers"], list), line_number

    question_key = f"confrag/train/{data['id']}"
    contexts, context_index_to_passage_key = validate_and_copy_contexts(
        websites=data["websites"],
        question_key=question_key,
        article_key_to_passage_key=article_key_to_passage_key,
    )
    answers = convert_answers(
        original_answers=data["answers"],
        context_index_to_passage_key=context_index_to_passage_key,
        question_key=question_key,
    )

    # Build the repository-wide QA record with task-specific annotations
    question = {
        "question_key": question_key,
        "question": data["question"],
        "answers": answers,
        "source": data["from"],
        "contradicts": data["contradicts"],
    }

    # Store the official webpage records separately from the QA record
    gold_contexts = {
        "question_key": question_key,
        "contexts": contexts,
    }
    return question, gold_contexts


def validate_and_copy_contexts(
    websites: list[dict[str, Any]],
    question_key: str,
    article_key_to_passage_key: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    contexts: list[dict[str, Any]] = []
    context_index_to_passage_key: dict[int, str] = {}
    seen_indices: set[int] = set()
    seen_passage_keys: set[str] = set()

    # Validate every official webpage field before selecting context metadata
    for website in websites:
        assert isinstance(website, dict), question_key
        assert set(website) == {
            "content",
            "answer",
            "reason",
            "trust_score",
            "index",
            "website",
        }, question_key
        assert isinstance(website["content"], str), question_key
        assert isinstance(website["answer"], str), question_key
        assert isinstance(website["reason"], list), question_key
        assert all(
            isinstance(reason, str)
            for reason in website["reason"]
        ), question_key
        assert isinstance(website["trust_score"], int), question_key
        assert 0 <= website["trust_score"] <= 10, question_key
        assert isinstance(website["index"], int), question_key
        assert isinstance(website["website"], str), question_key

        # Retain the official context index while validating references
        context_index = website["index"]
        if context_index in seen_indices:
            raise ValueError(
                f"Duplicate context index {context_index} in {question_key}"
            )
        seen_indices.add(context_index)

        # Assign one stable sequential key to each URL and text pair
        article_key = (website["website"], website["content"])
        if article_key not in article_key_to_passage_key:
            article_key_to_passage_key[article_key] = (
                f"confrag/passage#{len(article_key_to_passage_key):08d}"
            )
        passage_key = article_key_to_passage_key[article_key]
        context_index_to_passage_key[context_index] = passage_key

        # Keep each canonical article at most once within one gold context
        if passage_key in seen_passage_keys:
            continue
        seen_passage_keys.add(passage_key)

        # Exclude construction-time answer annotations to prevent gold leakage
        contexts.append(
            {
                "passage_key": passage_key,
                "text": website["content"],
                "website": website["website"],
                "trust_score": website["trust_score"],
            }
        )

    return contexts, context_index_to_passage_key


def convert_answers(
    original_answers: list[dict[str, Any]],
    context_index_to_passage_key: dict[int, str],
    question_key: str,
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    assigned_indices: set[int] = set()

    # Preserve every annotated viewpoint as one gold answer cluster
    for original_answer in original_answers:
        assert isinstance(original_answer, dict), question_key
        assert set(original_answer) == {
            "answer",
            "answer_judge_keyword",
            "index",
            "reason",
        }, question_key
        assert isinstance(original_answer["answer"], str), question_key
        assert isinstance(
            original_answer["answer_judge_keyword"],
            list,
        ), question_key
        assert all(
            isinstance(keyword, str)
            for keyword in original_answer["answer_judge_keyword"]
        ), question_key
        assert isinstance(original_answer["index"], list), question_key
        assert all(
            isinstance(context_index, int)
            for context_index in original_answer["index"]
        ), question_key
        assert isinstance(original_answer["reason"], list), question_key

        # Validate that the answer indices reference existing contexts and 
        # are mutually exclusive.
        answer_indices = original_answer["index"]
        if len(answer_indices) != len(set(answer_indices)):
            raise ValueError(
                f"Duplicate index within an answer cluster in {question_key}"
            )
        if not set(answer_indices).issubset(context_index_to_passage_key):
            raise ValueError(
                f"Unknown context index in an answer cluster in {question_key}"
            )
        overlapping_indices = assigned_indices.intersection(answer_indices)
        if overlapping_indices:
            raise ValueError(
                f"Contexts assigned to multiple answer clusters in "
                f"{question_key}: {sorted(overlapping_indices)}"
            )
        assigned_indices.update(answer_indices)

        reasons = convert_reasons(
            original_reasons=original_answer["reason"],
            answer_indices=set(answer_indices),
            question_key=question_key,
            context_index_to_passage_key=context_index_to_passage_key,
        )
        answers.append(
            {
                "answer": original_answer["answer"],
                "answer_type": "confrag",
                "passage_keys": list(dict.fromkeys(
                    context_index_to_passage_key[context_index]
                    for context_index in answer_indices
                )),
                "answer_judge_keywords": original_answer[
                    "answer_judge_keyword"
                ],
                "reasons": reasons,
            }
        )

    return answers


def convert_reasons(
    original_reasons: list[dict[str, Any]],
    answer_indices: set[int],
    question_key: str,
    context_index_to_passage_key: dict[int, str],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []

    # Convert every gold reason while retaining its supporting passage keys
    for original_reason in original_reasons:
        assert isinstance(original_reason, dict), question_key
        assert set(original_reason) == {
            "explain",
            "index",
            "reason_judge_keyword",
        }, question_key
        assert isinstance(original_reason["explain"], str), question_key
        assert isinstance(original_reason["index"], list), question_key
        assert all(
            isinstance(context_index, int)
            for context_index in original_reason["index"]
        ), question_key
        assert isinstance(
            original_reason["reason_judge_keyword"],
            list,
        ), question_key
        assert all(
            isinstance(keyword, str)
            for keyword in original_reason["reason_judge_keyword"]
        ), question_key

        # Validate that the reason indices are within the answer cluster
        reason_indices = original_reason["index"]
        if not set(reason_indices).issubset(answer_indices):
            raise ValueError(
                f"Reason cites a context outside its answer cluster in "
                f"{question_key}"
            )

        # Convert the official reason annotation into the repository-wide format
        reasons.append({
            "reason": original_reason["explain"],
            "passage_keys": list(dict.fromkeys(
                context_index_to_passage_key[context_index]
                for context_index in reason_indices
            )),
            "reason_judge_keywords": original_reason[
                "reason_judge_keyword"
            ],
        })

    return reasons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_articles_file",
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
