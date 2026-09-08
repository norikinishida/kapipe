import argparse
import gzip
import os
import pickle
from typing import Any

from tqdm import tqdm

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    input_modc_file: str = args.input_modc_file
    input_humc_file: str = args.input_humc_file
    output_articles_file: str = args.output_articles_file
    output_dir: str = args.output_dir

    # Require both official CONFACT splits before starting conversion
    for input_file in [input_modc_file, input_humc_file]:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Missing a CONFACT input file: {input_file}")

    # Load the official gzip-compressed pickle files without permitting classes
    original_modc = load_confact_file(input_file=input_modc_file)
    original_humc = load_confact_file(input_file=input_humc_file)

    # Assign one shared sequential key to each URL and text pair
    article_key_to_passage_key: dict[tuple[str, str], str] = {}

    # Convert both overlapping CONFACT evaluation splits independently
    modc_questions, modc_gold_contexts = convert_split(
        original_data=original_modc,
        split="modc",
        article_key_to_passage_key=article_key_to_passage_key,
    )
    humc_questions, humc_gold_contexts = convert_split(
        original_data=original_humc,
        split="humc",
        article_key_to_passage_key=article_key_to_passage_key,
    )

    # Ensure shared passage keys denote exactly the same released passages
    validate_shared_passages(
        modc_gold_contexts=modc_gold_contexts,
        humc_gold_contexts=humc_gold_contexts,
    )

    # Keep exactly one ModC corpus article for each assigned passage key
    passage_key_to_article: dict[str, dict[str, str]] = {}
    for contexts_for_question in modc_gold_contexts:
        for context in contexts_for_question["contexts"]:
            passage_key = context["passage_key"]
            if passage_key in passage_key_to_article:
                if context != passage_key_to_article[passage_key]:
                    raise ValueError(
                        f"Conflicting CONFACT article: {passage_key}"
                    )
            else:
                passage_key_to_article[passage_key] = context

    articles = list(passage_key_to_article.values())
    if len(articles) != len(article_key_to_passage_key):
        raise ValueError("Incomplete CONFACT article mapping")

    # Create the common CONFACT output directories
    utils.mkdir(output_dir)
    utils.mkdir(os.path.dirname(output_articles_file))

    # Save the complete released webpages as the shared retrieval corpus
    utils.write_jsonl(
        output_articles_file,
        articles,
        ensure_ascii=False,
    )
    print(
        f"Processed and saved {len(articles)} CONFACT articles into "
        f"{output_articles_file}"
    )

    # Save each split using the repository-wide QA filename convention
    for split, questions, gold_contexts in [
        ("modc", modc_questions, modc_gold_contexts),
        ("humc", humc_questions, humc_gold_contexts),
    ]:
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
            f"Processed and saved {len(questions)} {split} questions into "
            f"{output_questions_file}"
        )
        print(
            f"Processed and saved {len(gold_contexts)} {split} "
            f"gold-context instances into {output_gold_contexts_file}"
        )


def load_confact_file(input_file: str) -> list[dict[str, Any]]:
    # Decompress the official file while blocking arbitrary class construction
    with gzip.open(input_file, "rb") as file:
        data = RestrictedUnpickler(file).load()

    # Require the released top-level list structure
    assert isinstance(data, list), input_file
    assert all(isinstance(item, dict) for item in data), input_file
    return data


def convert_split(
    original_data: list[dict[str, Any]],
    split: str,
    article_key_to_passage_key: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert split in {"modc", "humc"}

    questions: list[dict[str, Any]] = []
    gold_contexts: list[dict[str, Any]] = []
    seen_question_keys: set[str] = set()
    seen_evidence_ids: set[str] = set()

    # Convert every claim and its question-scoped retrieved evidence
    for data in tqdm(original_data, desc=f"Processing CONFACT {split}"):
        validate_question_record(data=data, split=split)

        question_key = f"confact/{split}/{data['id']}"
        if question_key in seen_question_keys:
            raise ValueError(f"Duplicate CONFACT question key: {question_key}")
        seen_question_keys.add(question_key)

        # Normalize the known trailing whitespace in some Supported labels
        claim_label = data["label"].strip()
        label_to_answer = {
            "Supported": "yes",
            "Refuted": "no",
        }
        if claim_label not in label_to_answer:
            raise ValueError(
                f"Unexpected claim label in {question_key}: {claim_label}"
            )

        # Preserve claim-level provenance as QA metadata
        question = {
            "question_key": question_key,
            "question": data["question"],
            "answers": [
                {
                    "answer": label_to_answer[claim_label],
                }
            ],
            "claim": data["claim"],
            "claim_label": claim_label,
            "claim_date": data["claim_date"],
            "review_date": data["review_date"],
            "original_claim_url": data["original_claim_url"],
            "fact_checking_article": data["fact_checking_article"],
        }

        # Retain only model-facing passage content, identity, and source URL
        contexts = convert_evidence(
            evidence_list=data["evidence_url"],
            question_id=data["id"],
            question_key=question_key,
            split=split,
            seen_evidence_ids=seen_evidence_ids,
            article_key_to_passage_key=article_key_to_passage_key,
        )
        contexts_for_question = {
            "question_key": question_key,
            "contexts": contexts,
        }

        questions.append(question)
        gold_contexts.append(contexts_for_question)

    return questions, gold_contexts


def validate_question_record(
    data: dict[str, Any],
    split: str,
) -> None:
    # Detect changes to the official CONFACT claim schema immediately
    assert set(data) == {
        "id",
        "claim",
        "label",
        "claim_date",
        "review_date",
        "country",
        "question",
        "original_claim_url",
        "fact_checking_article",
        "evidence_url",
    }, split

    # Validate every source field used by or intentionally omitted from output
    assert isinstance(data["id"], int), split
    assert isinstance(data["claim"], str), data["id"]
    assert isinstance(data["label"], str), data["id"]
    assert isinstance(data["claim_date"], (str, type(None))), data["id"]
    assert isinstance(data["review_date"], (str, type(None))), data["id"]
    assert isinstance(data["country"], (str, type(None))), data["id"]
    assert isinstance(data["question"], str), data["id"]
    assert isinstance(
        data["original_claim_url"],
        (str, type(None)),
    ), data["id"]
    assert isinstance(data["fact_checking_article"], str), data["id"]
    assert isinstance(data["evidence_url"], list), data["id"]


def convert_evidence(
    evidence_list: list[dict[str, Any]],
    question_id: int,
    question_key: str,
    split: str,
    seen_evidence_ids: set[str],
    article_key_to_passage_key: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    seen_context_passage_keys: set[str] = set()

    # Convert every non-empty released webpage to the common Passage format
    for evidence in evidence_list:
        validate_evidence_record(
            evidence=evidence,
            question_key=question_key,
            split=split,
        )

        original_evidence_id = evidence["evidence_id"]
        if not original_evidence_id.startswith(f"{question_id}_"):
            raise ValueError(
                f"Evidence ID does not belong to {question_key}: "
                f"{original_evidence_id}"
            )
        if original_evidence_id in seen_evidence_ids:
            raise ValueError(
                f"Duplicate evidence ID within CONFACT {split}: "
                f"{original_evidence_id}"
            )
        seen_evidence_ids.add(original_evidence_id)

        # Exclude failed scrapes because an empty string is not a passage
        if not evidence["content"].strip():
            continue

        # Assign one stable sequential key to each URL and text pair
        article_key = (evidence["original_link"], evidence["content"])
        if article_key not in article_key_to_passage_key:
            article_key_to_passage_key[article_key] = (
                f"confact/passage#{len(article_key_to_passage_key):08d}"
            )
        passage_key = article_key_to_passage_key[article_key]

        # Keep each canonical article at most once within one gold context
        if passage_key in seen_context_passage_keys:
            continue
        seen_context_passage_keys.add(passage_key)

        contexts.append(
            {
                "passage_key": passage_key,
                "text": evidence["content"],
                "original_link": evidence["original_link"],
            }
        )

    return contexts


def validate_evidence_record(
    evidence: dict[str, Any],
    question_key: str,
    split: str,
) -> None:
    assert isinstance(evidence, dict), question_key

    # Require the split-specific released evidence schema
    if split == "modc":
        evidence_fields = set(evidence)
        required_evidence_fields = {
            "original_link",
            "content",
            "url_llm_check_result",
            "majority_vote",
            "html_file",
            "evidence_id",
        }
        optional_evidence_fields = {
            "archive_url",
            "text_noJustification_llm_check_result",
            "text_wJustification_llm_check_result",
        }
        assert required_evidence_fields.issubset(evidence_fields), question_key
        assert evidence_fields.issubset(
            required_evidence_fields | optional_evidence_fields
        ), question_key
    else:
        assert split == "humc"
        assert set(evidence) == {
            "evidence_id",
            "original_link",
            "content",
        }, question_key

    # Validate only the fields retained in the common Passage representation
    assert isinstance(evidence["evidence_id"], str), question_key
    assert isinstance(evidence["original_link"], str), question_key
    assert isinstance(evidence["content"], str), question_key


def validate_shared_passages(
    modc_gold_contexts: list[dict[str, Any]],
    humc_gold_contexts: list[dict[str, Any]],
) -> None:
    # Index ModC passages because HumC is an overlapping subset of ModC
    modc_passage_key_to_passage = {
        passage["passage_key"]: passage
        for contexts_for_question in modc_gold_contexts
        for passage in contexts_for_question["contexts"]
    }

    # Require every emitted HumC passage to match its shared ModC identity
    for contexts_for_question in humc_gold_contexts:
        for passage in contexts_for_question["contexts"]:
            passage_key = passage["passage_key"]
            if passage_key not in modc_passage_key_to_passage:
                raise ValueError(
                    f"HumC evidence is absent from ModC: {passage_key}"
                )
            if passage != modc_passage_key_to_passage[passage_key]:
                raise ValueError(
                    f"Shared passage key has different passages: "
                    f"{passage_key}"
                )


class RestrictedUnpickler(pickle.Unpickler):
    """Literal-only unpickler that rejects all global class lookups."""

    def find_class(self, module: str, name: str) -> Any:
        # Reject executable objects because CONFACT contains plain containers
        raise pickle.UnpicklingError(
            f"Forbidden global in CONFACT pickle: {module}.{name}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare CONFACT in the repository QA format."
    )
    parser.add_argument("--input_modc_file", type=str, required=True)
    parser.add_argument("--input_humc_file", type=str, required=True)
    parser.add_argument("--output_articles_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
