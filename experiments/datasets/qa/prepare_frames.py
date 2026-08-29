import argparse
import ast
import csv
import json
import os
import unicodedata
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import tensorflow_datasets as tfds
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    input_questions_file: str = args.input_questions_file
    wikipedia_config: str = args.wikipedia_config
    output_articles_file: str = args.output_articles_file
    output_questions_file: str = args.output_questions_file
    output_gold_contexts_file: str = args.output_gold_contexts_file

    # Validate the official annotations before loading the large corpus
    if not os.path.isfile(input_questions_file):
        raise FileNotFoundError(
            f"Missing the FRAMES question file: {input_questions_file}"
        )

    # Convert the official questions and collect every required Wikipedia title
    questions, evidence_lists = load_questions(
        input_questions_file=input_questions_file,
    )
    required_titles = {
        evidence["title"]
        for evidence_list in evidence_lists
        for evidence in evidence_list
    }

    # Prepare the full Wikipedia corpus and retain only the gold articles in memory
    title_to_article = prepare_wikipedia_articles(
        wikipedia_config=wikipedia_config,
        output_articles_file=output_articles_file,
        required_titles=required_titles,
    )

    # Preserve every question even when its historical gold article is absent
    missing_evidence_count = 0
    incomplete_question_count = 0
    for question, evidence_list in zip(
        questions,
        evidence_lists,
        strict=True,
    ):
        missing_titles = [
            evidence["title"]
            for evidence in evidence_list
            if evidence["title"] not in title_to_article
        ]
        question["full_found_in_wikipedia"] = not missing_titles

        if missing_titles:
            missing_evidence_count += len(missing_titles)
            incomplete_question_count += 1
            print(
                f"Incomplete question {question['question_key']}: "
                f"missing articles {missing_titles}"
            )

    # Build article-level gold contexts in the official evidence order
    gold_contexts = build_gold_contexts(
        questions=questions,
        evidence_lists=evidence_lists,
        title_to_article=title_to_article,
    )

    # Create the output directories before writing the converted annotations
    os.makedirs(
        os.path.dirname(output_questions_file),
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(output_gold_contexts_file),
        exist_ok=True,
    )

    # Save the common KAPipe QA and gold-context representations
    with open(output_questions_file, "w", encoding="utf-8") as file:
        json.dump(
            questions,
            file,
            ensure_ascii=False,
            indent=4,
        )
    with open(output_gold_contexts_file, "w", encoding="utf-8") as file:
        json.dump(
            gold_contexts,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # Report completeness separately from the number of retained questions
    print(
        f"Found {missing_evidence_count} missing evidence articles for "
        f"{incomplete_question_count} of {len(questions)} questions"
    )
    print(
        f"Processed and saved {len(questions)} questions into "
        f"{output_questions_file}"
    )
    print(
        f"Processed and saved {len(gold_contexts)} gold-context instances "
        f"into {output_gold_contexts_file}"
    )


def load_questions(
    input_questions_file: str,
) -> tuple[list[dict[str, Any]], list[list[dict[str, str]]]]:
    questions: list[dict[str, Any]] = []
    evidence_lists: list[list[dict[str, str]]] = []

    # Read multiline prompts correctly by letting the TSV parser handle quoting
    with open(
        input_questions_file,
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")
        required_field_names = {
            "Prompt",
            "Answer",
            "reasoning_types",
            "wiki_links",
        }
        assert reader.fieldnames is not None
        assert required_field_names.issubset(reader.fieldnames)

        for question_index, data in enumerate(
            tqdm(reader, desc="Processing FRAMES questions")
        ):
            # Validate the released fields used by the conversion
            prompt = data["Prompt"]
            answer = data["Answer"]
            reasoning_types_text = data["reasoning_types"]
            wiki_links_text = data["wiki_links"]
            assert isinstance(prompt, str) and prompt.strip()
            assert isinstance(answer, str) and answer.strip()
            assert isinstance(reasoning_types_text, str)
            assert isinstance(wiki_links_text, str)

            # Parse the Python-list representation used by the official TSV
            wiki_links = ast.literal_eval(wiki_links_text)
            assert isinstance(wiki_links, list)
            assert all(isinstance(wiki_link, str) for wiki_link in wiki_links)

            # Normalize and deduplicate evidence while preserving source order
            evidence_list: list[dict[str, str]] = []
            seen_titles: set[str] = set()
            for wiki_link in wiki_links:
                title = normalize_wikipedia_title(wikipedia_url=wiki_link)
                if title in seen_titles:
                    continue

                seen_titles.add(title)
                evidence_list.append(
                    {
                        "url": wiki_link.strip(),
                        "title": title,
                    }
                )

            # Split the official multi-label annotation into individual labels
            reasoning_types = [
                reasoning_type.strip()
                for reasoning_type in reasoning_types_text.split("|")
                if reasoning_type.strip()
            ]

            question_key = f"frames-test-{question_index}"
            questions.append(
                {
                    "question_key": question_key,
                    "question": prompt.strip(),
                    "answers": [
                        {
                            "answer": answer.strip(),
                        }
                    ],
                    "reasoning_types": reasoning_types,
                    "evidence_list": [
                        {
                            "url": evidence["url"],
                            "title": evidence["title"],
                        }
                        for evidence in evidence_list
                    ],
                }
            )
            evidence_lists.append(evidence_list)

    return questions, evidence_lists


def normalize_wikipedia_title(wikipedia_url: str) -> str:
    # Remove surrounding whitespace before parsing malformed source URLs
    normalized_url = wikipedia_url.strip()

    # Add the scheme omitted from two URLs in the official FRAMES annotations
    if normalized_url.startswith(("en.wikipedia.org/", "en.m.wikipedia.org/")):
        normalized_url = f"https://{normalized_url}"

    # Replace the only shortened URL with its fixed Wikipedia redirect target
    if normalized_url == "https://w.wiki/ASFv":
        normalized_url = "https://en.wikipedia.org/wiki/Harry_C._Bradley_(actor)"

    parsed_url = urlsplit(normalized_url)

    # Restrict evidence to the Wikipedia domains present in the official data
    assert parsed_url.scheme in {"http", "https"}
    assert parsed_url.netloc.lower() in {
        "en.wikipedia.org",
        "en.m.wikipedia.org",
        "simple.wikipedia.org",
    }

    # Extract titles from either an article path or a search/redirect URL
    if parsed_url.path.startswith("/wiki/"):
        encoded_title = parsed_url.path[len("/wiki/"):]
    else:
        assert parsed_url.path == "/w/index.php"
        query = parse_qs(parsed_url.query)
        if "search" in query:
            assert len(query["search"]) == 1
            encoded_title = query["search"][0]
        else:
            assert len(query["title"]) == 1
            encoded_title = query["title"][0]

    # Decode the article title and normalize its display form
    title = unquote(encoded_title).replace("_", " ").strip()
    title = unicodedata.normalize("NFC", title)
    assert title
    return title


def load_evidence_articles(
    input_articles_file: str,
    required_titles: set[str],
) -> dict[str, dict[str, str]]:
    title_to_article: dict[str, dict[str, str]] = {}

    # Stream the existing full corpus and retain only FRAMES gold articles
    with open(input_articles_file, encoding="utf-8") as file:
        for line in tqdm(file, desc="Finding FRAMES evidence articles"):
            article = json.loads(line)
            assert isinstance(article, dict)
            assert isinstance(article["title"], str)
            assert isinstance(article["text"], str)

            title = article["title"]
            if title not in required_titles:
                continue

            assert title not in title_to_article
            title_to_article[title] = {
                "title": title,
                "text": article["text"],
            }

    return title_to_article


def prepare_wikipedia_articles(
    wikipedia_config: str,
    output_articles_file: str,
    required_titles: set[str],
) -> dict[str, dict[str, str]]:
    title_to_article: dict[str, dict[str, str]] = {}

    # Read the exact prepared snapshot because its original raw dump is unavailable
    wikipedia = tfds.load(
        f"wikipedia/{wikipedia_config}:1.0.0",
        data_dir="gs://tfds-data/datasets",
        download=False,
        split="train",
        shuffle_files=False,
    )

    # Create the corpus directory before streaming the complete snapshot
    os.makedirs(
        os.path.dirname(output_articles_file),
        exist_ok=True,
    )
    temporary_output_file = f"{output_articles_file}.tmp"

    # Write atomically so an interrupted run is not mistaken for a full corpus
    with open(temporary_output_file, "w", encoding="utf-8") as file:
        for original_article in tqdm(
            tfds.as_numpy(wikipedia),
            desc=f"Preparing Wikipedia {wikipedia_config}",
        ):
            title = decode_tfds_text(original_article["title"])
            text = decode_tfds_text(original_article["text"])
            article = {
                "title": title,
                "text": text,
            }
            file.write(json.dumps(article, ensure_ascii=False) + "\n")

            # Retain only gold articles in memory while writing the full corpus
            if title in required_titles:
                assert title not in title_to_article
                title_to_article[title] = article

    # Publish the corpus only after every TFDS article has been written
    os.replace(temporary_output_file, output_articles_file)
    print(f"Saved Wikipedia articles into {output_articles_file}")
    return title_to_article


def decode_tfds_text(value: Any) -> str:
    # Decode byte strings emitted by tfds.as_numpy without altering text
    if isinstance(value, bytes):
        return value.decode("utf-8")

    # Accept native strings for compatibility across TFDS versions
    assert isinstance(value, str)
    return value


def build_gold_contexts(
    questions: list[dict[str, Any]],
    evidence_lists: list[list[dict[str, str]]],
    title_to_article: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    assert len(questions) == len(evidence_lists)
    gold_contexts: list[dict[str, Any]] = []

    for question, evidence_list in zip(
        questions,
        evidence_lists,
        strict=True,
    ):
        contexts: list[dict[str, Any]] = []

        # Preserve unavailable evidence with its normalized title and an empty body
        for evidence in evidence_list:
            title = evidence["title"]
            if title in title_to_article:
                contexts.append(
                    {
                        "title": title,
                        "text": title_to_article[title]["text"],
                        "found_in_wikipedia": True,
                    }
                )
            else:
                contexts.append(
                    {
                        "title": title,
                        "text": "",
                        "found_in_wikipedia": False,
                    }
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
        "--input_questions_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wikipedia_config",
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
