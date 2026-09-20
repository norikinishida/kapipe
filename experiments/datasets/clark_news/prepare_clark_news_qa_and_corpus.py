import argparse
import csv
import datetime
import json
import os
from typing import Any

from tqdm import tqdm

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    input_dir: str = args.input_dir
    output_articles_file: str = args.output_articles_file
    output_questions_file: str = args.output_questions_file
    output_gold_contexts_file: str = args.output_gold_contexts_file

    # Validate the three files distributed with the full CLARK-News dataset
    articles_file = os.path.join(input_dir, "external_sources.json")
    questions_file = os.path.join(input_dir, "questions.csv")
    facts_file = os.path.join(input_dir, "property_to_results.csv")
    for input_file in [questions_file, articles_file, facts_file]:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Missing an input file: {input_file}")

    # Index the annotated facts before linking questions and articles
    (
        url_and_timestamp_to_triples,
        relation_object_and_timestamp_to_subjects_and_urls,
    ) = load_fact_indices(facts_file=facts_file)

    # Extract the articles and assign stable keys to each URL and timestamp pair
    articles, url_and_timestamp_to_passage_key = load_articles(
        articles_file=articles_file,
        url_and_timestamp_to_triples=url_and_timestamp_to_triples,
    )

    # Extract the questions and answers
    questions = load_questions(
        questions_file=questions_file,
        relation_object_and_timestamp_to_subjects_and_urls=(
            relation_object_and_timestamp_to_subjects_and_urls
        ),
        url_and_timestamp_to_passage_key=url_and_timestamp_to_passage_key,
    )

    # Resolve annotated evidence articles into the gold-context format
    passage_key_to_article = {
        article["passage_key"]: article
        for article in articles
    }
    gold_contexts = build_gold_contexts(
        questions=questions,
        passage_key_to_article=passage_key_to_article,
    )

    # Remove intermediate passage keys duplicated in the gold contexts
    for question in questions:
        del question["evidence_passage_keys"]

    # Create all output directories before writing the converted dataset
    for output_file in [
        output_questions_file,
        output_gold_contexts_file,
        output_articles_file,
    ]:
        utils.mkdir(os.path.dirname(output_file))

    write_jsonl(
        output_file=output_articles_file,
        records=articles,
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
        f"Processed and saved {len(gold_contexts)} gold-context instances into "
        f"{output_gold_contexts_file}"
    )


def load_fact_indices(
    facts_file: str,
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str, str], list[tuple[str, str]]],
]:
    # Associate every source URL and date with the facts supported by it
    url_and_timestamp_to_triples: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    # Associate every relation, object, and date with candidate subjects and URLs
    relation_object_and_timestamp_to_subjects_and_urls: dict[
        tuple[str, str, str],
        list[tuple[str, str]],
    ] = {}

    with open(facts_file, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader, desc="Indexing CLARK-News facts"):
            # Validate all fields used to link facts, articles, and questions
            subject = row["subjectLabel"].strip()
            relation = row["propertyLabel"].strip()
            object_: str | float = row["objectLabel"].strip()
            if not object_:
                object_ = float("nan")
            url = row["selected_link"].strip()
            timestamp = normalize_date(row["source_date"])

            assert subject
            assert relation
            assert url
            assert timestamp is not None

            triple = {
                "subject": subject,
                "relation": relation,
                "object": object_,
                "timestamp": timestamp,
            }
            url_and_timestamp_to_triples.setdefault(
                (url, timestamp),
                [],
            ).append(triple)
            relation_object_and_timestamp_to_subjects_and_urls.setdefault(
                (relation, str(object_), timestamp),
                [],
            ).append((subject, url))

    return (
        url_and_timestamp_to_triples,
        relation_object_and_timestamp_to_subjects_and_urls,
    )


def load_articles(
    articles_file: str,
    url_and_timestamp_to_triples: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    # Load the URL-indexed external sources distributed by CLARK-News
    with open(articles_file, encoding="utf-8") as file:
        url_to_timestamp_to_article: dict[
            str,
            dict[str, dict[str, Any]],
        ] = json.load(file)

    articles: list[dict[str, Any]] = []
    url_and_timestamp_to_passage_key: dict[tuple[str, str], str] = {}

    # Preserve the insertion order used by the project080 notebook
    for url, timestamp_to_article in tqdm(
        url_to_timestamp_to_article.items(),
        desc="Converting CLARK-News articles",
    ):
        for original_article in timestamp_to_article.values():
            timestamp = normalize_date(original_article["source_timestamp"])
            assert timestamp is not None

            passage_key = f"clark_news/article#{len(articles):08d}"
            url_and_timestamp = (url, timestamp)
            assert url_and_timestamp not in url_and_timestamp_to_passage_key
            url_and_timestamp_to_passage_key[url_and_timestamp] = passage_key

            # Preserve the article and the temporal facts supported by it
            article = {
                "passage_key": passage_key,
                "text": original_article["source_text"],
                "timestamp": timestamp,
                "source": (
                    original_article["archive_url"]
                    if "archive_url" in original_article
                    else url
                ),
                "url": url,
                "triples": url_and_timestamp_to_triples.get(
                    url_and_timestamp,
                    [],
                ),
            }
            articles.append(article)

    return articles, url_and_timestamp_to_passage_key


def load_questions(
    questions_file: str,
    relation_object_and_timestamp_to_subjects_and_urls: dict[
        tuple[str, str, str],
        list[tuple[str, str]],
    ],
    url_and_timestamp_to_passage_key: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    # Group rows by question text and date because list answers occupy separate rows
    group_key_to_rows: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    with open(questions_file, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(
            tqdm(reader, desc="Linking CLARK-News questions")
        ):
            question_text = row["Question"].strip()
            answer = row["Answer"].strip()
            relation = row["Question type"].strip()

            # Prefer the known timestamp and retain the project080 fallback
            timestamp_value = row["Known start timestamp"].strip()
            if not timestamp_value:
                timestamp_value = row["Start timestamp"].strip()
            if not timestamp_value:
                timestamp_value = "2020-01-01"
            timestamp = normalize_date(timestamp_value)
            assert question_text
            assert answer
            assert relation
            assert timestamp is not None

            # Match the longest subject explicitly mentioned in the question
            candidates = (
                relation_object_and_timestamp_to_subjects_and_urls.get(
                    (relation, answer, timestamp),
                    [],
                )
            )
            matched_candidates = [
                (subject, url)
                for subject, url in candidates
                if subject.lower() in question_text.lower()
            ]

            triple: dict[str, str] | None = None
            evidence_passage_key: str | None = None
            if matched_candidates:
                best_subject = max(
                    (
                        subject
                        for subject, _ in matched_candidates
                    ),
                    key=len,
                )
                matched_urls = {
                    url
                    for subject, url in matched_candidates
                    if subject == best_subject
                }
                assert len(matched_urls) == 1
                matched_url = next(iter(matched_urls))
                passage_key = url_and_timestamp_to_passage_key[
                    (matched_url, timestamp)
                ]

                triple = {
                    "subject": best_subject,
                    "relation": relation,
                    "object": answer,
                    "timestamp": timestamp,
                }
                evidence_passage_key = passage_key

            row_record = {
                "question_key": f"question#{row_index:04d}",
                "answer": answer,
                "triple": triple,
                "evidence_passage_key": evidence_passage_key,
            }
            group_key_to_rows.setdefault(
                (question_text, timestamp),
                [],
            ).append(row_record)

    # Merge duplicate rows into one temporal question with one or more answers
    questions: list[dict[str, Any]] = []
    for (question_text, timestamp), rows in group_key_to_rows.items():
        answer_texts = sorted({row["answer"] for row in rows})
        question_key = "clark_news/" + "|".join(
            row["question_key"] for row in rows
        )
        time_agnostic_question_key = str(abs(hash(question_text)))

        triples = [row["triple"] for row in rows]
        evidence_passage_keys = [
            row["evidence_passage_key"]
            for row in rows
        ]

        question = {
            "question_key": question_key,
            "time_agnostic_question_key": time_agnostic_question_key,
            "question": question_text,
            "timestamp": timestamp,
            "answers": [
                {
                    "answer": answer_text,
                    "answer_type": "list",
                    "list_index": answer_index,
                }
                for answer_index, answer_text in enumerate(answer_texts)
            ],
            "triples": triples,
            "evidence_passage_keys": evidence_passage_keys,
        }
        questions.append(question)

    # Reproduce the project080 group and timestamp ordering
    question_text_to_questions: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        question_text_to_questions.setdefault(
            question["question"],
            [],
        ).append(question)

    question_groups = list(question_text_to_questions.values())
    for group in question_groups:
        group.sort(key=lambda question: question["timestamp"])
    question_groups.sort(key=len, reverse=True)

    questions = [
        question
        for group in question_groups
        for question in group
    ]
    return questions


def build_gold_contexts(
    questions: list[dict[str, Any]],
    passage_key_to_article: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gold_contexts: list[dict[str, Any]] = []

    # Resolve each evidence reference and preserve empty contexts
    for question in questions:
        contexts = [
            passage_key_to_article[evidence_passage_key]
            for evidence_passage_key in question["evidence_passage_keys"]
            if evidence_passage_key is not None
        ]
        gold_contexts.append(
            {
                "question_key": question["question_key"],
                "contexts": contexts,
            }
        )

    return gold_contexts


def normalize_date(value: str) -> str | None:
    # Treat blank values as missing timestamps
    normalized_value = value.strip()
    if not normalized_value:
        return None

    # Parse ISO 8601 timestamps used in questions and external sources
    try:
        iso_value = normalized_value.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(iso_value).date().isoformat()
    except ValueError:
        pass

    # Parse the US-style dates used in property_to_results.csv
    for date_format in ["%m/%d/%Y", "%m/%Y", "%Y-%m-%d"]:
        try:
            return datetime.datetime.strptime(
                normalized_value,
                date_format,
            ).date().isoformat()
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp: {value}")


def write_jsonl(
    output_file: str,
    records: list[dict[str, Any]],
) -> None:
    # Write one article per line to support streaming retrieval pipelines
    with open(output_file, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
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
