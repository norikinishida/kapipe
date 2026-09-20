import argparse
import json
import os
from typing import Any, TextIO
import unicodedata

from tqdm import tqdm

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    input_train_file: str = args.input_train_file
    input_dev_file: str = args.input_dev_file
    input_wikipedia_dir: str = args.input_wikipedia_dir
    output_articles_file: str = args.output_articles_file
    output_dir: str = args.output_dir

    # Validate that all required FEVER input files and directories exist
    for input_path in [
        input_train_file,
        input_dev_file,
        input_wikipedia_dir,
    ]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing a FEVER input: {input_path}")

    # Load the labelled splits and collect all referenced Wikipedia sentences
    claims_by_split: dict[str, list[dict[str, Any]]] = {}
    required_evidence_keys: set[tuple[str, int]] = set()
    for split, input_file in [
        ("train", input_train_file),
        ("dev", input_dev_file),
    ]:
        claims = load_claims(
            input_file=input_file,
            split=split,
            required_evidence_keys=required_evidence_keys,
        )
        claims_by_split[split] = claims

    # Create the output directories before writing the converted data
    utils.mkdir(output_dir)
    output_articles_dir = os.path.dirname(output_articles_file)
    if output_articles_dir != "":
        utils.mkdir(output_articles_dir)

    # Convert the official June 2017 Wikipedia dump and resolve gold sentences
    evidence_key_to_text = convert_wikipedia(
        input_wikipedia_dir=input_wikipedia_dir,
        output_articles_file=output_articles_file,
        required_evidence_keys=required_evidence_keys,
    )

    # Fail if any official evidence annotation cannot be resolved in the dump
    missing_evidence_keys = required_evidence_keys - set(evidence_key_to_text)
    if missing_evidence_keys:
        example_missing_keys = sorted(missing_evidence_keys)[:10]
        raise ValueError(
            f"Failed to resolve {len(missing_evidence_keys)} FEVER evidence "
            f"sentences. Examples: {example_missing_keys}"
        )

    # Convert and save each labelled split in the common QA representation
    for split, claims in claims_by_split.items():
        questions, gold_contexts_sets = convert_claims(
            claims=claims,
            split=split,
            evidence_key_to_text=evidence_key_to_text,
        )

        output_questions_file = os.path.join(output_dir, f"{split}.json")
        output_gold_contexts_sets_file = os.path.join(
            output_dir,
            f"{split}.gold_contexts_sets.json",
        )

        utils.write_json(
            output_questions_file,
            questions,
            ensure_ascii=False,
        )
        utils.write_json(
            output_gold_contexts_sets_file,
            gold_contexts_sets,
            ensure_ascii=False,
        )

        print(
            f"Processed and saved {len(questions)} FEVER {split} questions "
            f"into {output_questions_file}"
        )
        print(
            f"Processed and saved {len(gold_contexts_sets)} FEVER {split} "
            f"gold-context-set instances into "
            f"{output_gold_contexts_sets_file}"
        )


def load_claims(
    input_file: str,
    split: str,
    required_evidence_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen_claim_ids: set[int] = set()

    # Read one official claim per JSONL record without flattening evidence sets
    with open(input_file, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            data = json.loads(line)
            validate_claim(data=data, split=split, line_number=line_number)

            claim_id = data["id"]
            if claim_id in seen_claim_ids:
                raise ValueError(
                    f"Duplicate FEVER claim ID in {split}: {claim_id}"
                )
            seen_claim_ids.add(claim_id)

            # Collect every non-null sentence reference across alternative sets
            for evidence_set in data["evidence"]:
                for evidence in evidence_set:
                    wikipedia_id = evidence[2]
                    sentence_id = evidence[3]
                    if wikipedia_id is None:
                        continue
                    assert isinstance(sentence_id, int)
                    required_evidence_keys.add(
                        (
                            normalize_wikipedia_id(wikipedia_id),
                            sentence_id,
                        )
                    )

            claims.append(data)

    print(f"Loaded {len(claims)} FEVER {split} claims from {input_file}")
    return claims


def normalize_wikipedia_id(wikipedia_id: str) -> str:
    # Canonicalize equivalent accented characters used inconsistently by FEVER
    return unicodedata.normalize("NFC", wikipedia_id)


def validate_claim(
    data: dict[str, Any],
    split: str,
    line_number: int,
) -> None:
    # Detect changes to the official labelled FEVER schema immediately
    assert set(data) == {
        "id",
        "verifiable",
        "label",
        "claim",
        "evidence",
    }, (
        split,
        line_number,
        set(data),
    )
    assert isinstance(data["id"], int), (split, line_number)
    assert isinstance(data["verifiable"], str), (split, data["id"])
    assert data["verifiable"] in {
        "VERIFIABLE",
        "NOT VERIFIABLE",
    }, (split, data["id"], data["verifiable"])
    assert isinstance(data["label"], str), (split, data["id"])
    assert data["label"] in {
        "SUPPORTS",
        "REFUTES",
        "NOT ENOUGH INFO",
    }, (split, data["id"], data["label"])
    assert isinstance(data["claim"], str), (split, data["id"])
    assert isinstance(data["evidence"], list), (split, data["id"])
    assert all(
        isinstance(evidence_set, list)
        for evidence_set in data["evidence"]
    ), (split, data["id"])

    # Validate the nested official evidence tuples and their nullability
    for evidence_set in data["evidence"]:
        assert len(evidence_set) > 0, (split, data["id"])
        for evidence in evidence_set:
            assert isinstance(evidence, list), (split, data["id"])
            assert len(evidence) == 4, (split, data["id"], evidence)
            assert isinstance(evidence[0], int), (split, data["id"], evidence)
            assert isinstance(evidence[1], (int, type(None))), (
                split,
                data["id"],
                evidence,
            )
            assert isinstance(evidence[2], (str, type(None))), (
                split,
                data["id"],
                evidence,
            )
            assert isinstance(evidence[3], (int, type(None))), (
                split,
                data["id"],
                evidence,
            )
            assert (evidence[2] is None) == (evidence[3] is None), (
                split,
                data["id"],
                evidence,
            )

    # Validate that verifiable claims have non-null evidence and 
    # that non-verifiable claims do not.
    has_non_null_evidence = any(
        evidence[2] is not None
        for evidence_set in data["evidence"]
        for evidence in evidence_set
    )
    if data["label"] == "NOT ENOUGH INFO":
        assert data["verifiable"] == "NOT VERIFIABLE", (
            split,
            data["id"],
        )
        assert not has_non_null_evidence, (split, data["id"])
    else:
        assert data["verifiable"] == "VERIFIABLE", (split, data["id"])
        assert has_non_null_evidence, (split, data["id"])


def convert_wikipedia(
    input_wikipedia_dir: str,
    output_articles_file: str,
    required_evidence_keys: set[tuple[str, int]],
) -> dict[tuple[str, int], str]:
    wikipedia_files = [
        os.path.join(input_wikipedia_dir, filename)
        for filename in sorted(os.listdir(input_wikipedia_dir))
        if filename.endswith(".jsonl")
    ]
    if not wikipedia_files:
        raise FileNotFoundError(
            f"No FEVER Wikipedia JSONL files found in {input_wikipedia_dir}"
        )

    required_sentence_ids_by_wikipedia_id: dict[str, set[int]] = {}
    for wikipedia_id, sentence_id in required_evidence_keys:
        if wikipedia_id not in required_sentence_ids_by_wikipedia_id:
            required_sentence_ids_by_wikipedia_id[wikipedia_id] = set()
        required_sentence_ids_by_wikipedia_id[wikipedia_id].add(sentence_id)

    evidence_key_to_text: dict[tuple[str, int], str] = {}
    article_count = 0

    # Stream the large official dump to avoid retaining the corpus in memory
    with open(output_articles_file, "w", encoding="utf-8") as output_file:
        for input_file in tqdm(
            wikipedia_files,
            desc="Processing FEVER Wikipedia files",
        ):
            article_count += convert_wikipedia_file(
                input_file=input_file,
                output_file=output_file,
                required_sentence_ids_by_wikipedia_id=(
                    required_sentence_ids_by_wikipedia_id
                ),
                evidence_key_to_text=evidence_key_to_text,
            )

    print(
        f"Processed and saved {article_count} FEVER Wikipedia articles into "
        f"{output_articles_file}"
    )
    return evidence_key_to_text


def convert_wikipedia_file(
    input_file: str,
    output_file: TextIO,
    required_sentence_ids_by_wikipedia_id: dict[str, set[int]],
    evidence_key_to_text: dict[tuple[str, int], str],
) -> int:
    article_count = 0

    # Convert every page in one official Wikipedia shard
    with open(input_file, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            article = json.loads(line)
            assert set(article) == {"id", "text", "lines"}, (
                input_file,
                line_number,
                set(article),
            )
            assert isinstance(article["id"], str), (input_file, line_number)
            assert isinstance(article["text"], str), (input_file, line_number)
            assert isinstance(article["lines"], str), (input_file, line_number)

            # Exclude pages without article text from the retrieval corpus
            if article["text"].strip() == "":
                continue

            wikipedia_id = normalize_wikipedia_id(article["id"])

            # Preserve the official page ID while exposing a readable title
            output_article = {
                "passage_key": f"wikipedia/2017-06-01/{wikipedia_id}",
                "title": wikipedia_id.replace("_", " "),
                "text": article["text"],
                "wikipedia_id": wikipedia_id,
            }
            output_file.write(
                json.dumps(output_article, ensure_ascii=False) + "\n"
            )
            article_count += 1

            # Parse sentence lines only for pages referenced by gold evidence
            if wikipedia_id not in required_sentence_ids_by_wikipedia_id:
                continue
            sentence_id_to_text = parse_wikipedia_lines(
                lines=article["lines"],
                wikipedia_id=wikipedia_id,
            )
            for sentence_id in required_sentence_ids_by_wikipedia_id[
                wikipedia_id
            ]:
                if sentence_id not in sentence_id_to_text:
                    continue
                evidence_key_to_text[(wikipedia_id, sentence_id)] = (
                    sentence_id_to_text[sentence_id]
                )

    return article_count


def parse_wikipedia_lines(
    lines: str,
    wikipedia_id: str,
) -> dict[int, str]:
    sentence_id_to_text: dict[int, str] = {}

    # Extract the sentence ID and text before the optional hyperlink columns
    for raw_line in lines.splitlines():
        fields = raw_line.split("\t")

        # Skip hyperlink-only continuation lines without a sentence ID
        if not fields[0].isdigit():
            continue

        # Validate that each line contains at least a sentence ID and text
        if len(fields) < 2:
            raise ValueError(
                f"Malformed FEVER Wikipedia line in {wikipedia_id}: {raw_line}"
            )

        # Extract the sentence ID and text from the first two columns
        sentence_id = int(fields[0])
        sentence_text = fields[1]

        # Validate that the sentence ID is unique within the article
        if sentence_id in sentence_id_to_text:
            raise ValueError(
                f"Duplicate sentence ID in {wikipedia_id}: {sentence_id}"
            )
        sentence_id_to_text[sentence_id] = sentence_text

    return sentence_id_to_text


def convert_claims(
    claims: list[dict[str, Any]],
    split: str,
    evidence_key_to_text: dict[tuple[str, int], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions: list[dict[str, Any]] = []
    gold_contexts_sets: list[dict[str, Any]] = []

    # Convert every claim while preserving each alternative evidence set
    for data in tqdm(claims, desc=f"Converting FEVER {split}"):
        question_key = f"fever/{split}/{data['id']}"
        claim = data["claim"]
        question_text = (
            "Based on the available evidence, should the following claim be "
            "classified as SUPPORTS, REFUTES, or NOT ENOUGH INFO?\n"
            f"Claim: {claim}"
        )

        questions.append(
            {
                "question_key": question_key,
                "claim": claim,
                "question": question_text,
                "candidate_answers": [
                    "SUPPORTS",
                    "REFUTES",
                    "NOT ENOUGH INFO",
                ],
                "answers": [
                    {
                        "answer": data["label"],
                    }
                ],
                "verifiable": data["verifiable"],
            }
        )

        contexts_sets: list[dict[str, list[dict[str, Any]]]] = []
        if data["label"] != "NOT ENOUGH INFO":
            for evidence_set in data["evidence"]:
                contexts: list[dict[str, Any]] = []
                for evidence in evidence_set:
                    wikipedia_id = evidence[2]
                    sentence_id = evidence[3]
                    assert isinstance(wikipedia_id, str), (
                        question_key,
                        evidence,
                    )
                    assert isinstance(sentence_id, int), (
                        question_key,
                        evidence,
                    )

                    # Match the canonical Wikipedia ID used by the corpus
                    normalized_wikipedia_id = normalize_wikipedia_id(
                        wikipedia_id
                    )
                    contexts.append(
                        {
                            "passage_key": (
                                "wikipedia/2017-06-01/"
                                f"{normalized_wikipedia_id}"
                                f"/sentence#{sentence_id:04d}"
                            ),
                            "title": normalized_wikipedia_id.replace("_", " "),
                            "text": evidence_key_to_text[
                                (normalized_wikipedia_id, sentence_id)
                            ],
                            "wikipedia_id": normalized_wikipedia_id,
                            "sentence_id": sentence_id,
                        }
                    )
                contexts_sets.append({"contexts": contexts})

        gold_contexts_sets.append(
            {
                "question_key": question_key,
                "contexts_sets": contexts_sets,
            }
        )

    return questions, gold_contexts_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_train_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_dev_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_wikipedia_dir",
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
