import argparse
import datetime
import gzip
import importlib.util
import json
import os
from collections.abc import Iterator
from types import ModuleType
from typing import Any

from tqdm import tqdm

from kapipe import utils


QA_TOTALS: dict[str, int] = {
    "train": 99_402,
    "valid": 9_939,
    "eval": 36_378,
}


def main(args: argparse.Namespace) -> None:
    input_dir: str = args.input_dir
    wmt_dir: str = args.wmt_dir
    extraction_file: str = args.extraction_file
    output_questions_dir: str = args.output_questions_dir
    output_articles_file: str = args.output_articles_file

    # Map the official valid and eval splits to the standard dev and test names
    source_split_to_output_split = {
        "train": "train",
        "valid": "dev",
        "eval": "test",
    }
    output_split_to_questions: dict[str, list[dict[str, Any]]] = {}
    required_article_ids: set[str] = set()

    # Normalize the official QA files and create KAPipe question records
    for source_split, output_split in source_split_to_output_split.items():
        input_file = os.path.join(
            input_dir,
            f"streaminqa_{source_split}.jsonl.gz",
        )
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Missing a StreamingQA split: {input_file}")

        qas = load_qas(
            input_file=input_file,
            source_split=source_split,
            output_split=output_split,
        )
        output_qas_file = os.path.join(
            input_dir,
            f"qas.{output_split}.jsonl",
        )
        write_jsonl_records(
            output_file=output_qas_file,
            records=qas,
        )

        questions = convert_questions(
            qas=qas,
            output_split=output_split,
        )
        output_split_to_questions[output_split] = questions
        required_article_ids.update(
            evidence_article_id
            for question in questions
            for evidence_article_id in question["evidence_article_ids"]
        )

        print(
            f"Processed and saved {len(qas)} normalized QA records into "
            f"{output_qas_file}"
        )

    # Load the official extraction module at its explicit repository path
    extraction_module = load_extraction_module(
        extraction_file=extraction_file,
    )

    # Create the normalized docs and the complete article file
    evidence_articles = write_docs_and_articles(
        input_dir=input_dir,
        wmt_dir=wmt_dir,
        extraction_module=extraction_module,
        output_docs_file=os.path.join(input_dir, "docs.jsonl"),
        output_articles_file=output_articles_file,
        required_article_ids=required_article_ids,
    )

    # Verify all QA evidence references before writing the final question files
    missing_article_ids = required_article_ids - set(evidence_articles)
    if missing_article_ids:
        examples = sorted(missing_article_ids)[:10]
        raise ValueError(f"Missing StreamingQA evidence articles: {examples}")

    # Save each complete question split and its gold contexts
    utils.mkdir(output_questions_dir)
    for output_split in ["train", "dev", "test"]:
        questions = output_split_to_questions[output_split]
        gold_contexts = build_gold_contexts(
            questions=questions,
            evidence_articles=evidence_articles,
        )

        # Remove intermediate article IDs duplicated in the gold contexts
        for question in questions:
            del question["evidence_article_ids"]

        output_questions_file = os.path.join(
            output_questions_dir,
            f"{output_split}.json",
        )
        output_gold_contexts_file = os.path.join(
            output_questions_dir,
            f"{output_split}.gold_contexts.json",
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


def load_qas(
    input_file: str,
    source_split: str,
    output_split: str,
) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []

    # Preserve official QA fields while normalizing timestamps and split names
    for record in tqdm(
        read_gzip_jsonl(input_file=input_file),
        total=QA_TOTALS[source_split],
        desc=f"Normalizing StreamingQA {output_split}",
    ):
        assert isinstance(record["qa_id"], str)
        assert isinstance(record["question"], str)
        assert isinstance(record["answers"], list)
        assert isinstance(record["answers_additional"], list)
        assert isinstance(record["question_ts"], int)
        assert isinstance(record["evidence_ts"], int)
        assert isinstance(record["evidence_id"], str)

        qa = {
            "qa_id": record["qa_id"],
            "question": record["question"],
            "question_ts": record["question_ts"],
            "question_date": timestamp_to_date(record["question_ts"]),
            "answers": record["answers"],
            "answers_additional": record["answers_additional"],
            "evidence_doc_id": record["evidence_id"],
            "evidence_ts": record["evidence_ts"],
            "evidence_date": timestamp_to_date(record["evidence_ts"]),
            "metadata": {
                "dataset": "StreamingQA",
                "split": output_split,
                "recent_or_past": record["recent_or_past"],
                "written_or_generated": record["written_or_generated"],
                "toxicity": {
                    "identity_attack": record["toxicity_identity_attack"],
                    "insult": record["toxicity_insult"],
                    "profanity": record["toxicity_profanity"],
                    "severe_toxicity": record["toxicity_severe_toxicity"],
                    "sexually_explicit": record[
                        "toxicity_sexually_explicit"
                    ],
                    "threat": record["toxicity_threat"],
                },
            },
        }
        qas.append(qa)

    return qas


def convert_questions(
    qas: list[dict[str, Any]],
    output_split: str,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    # Convert normalized QA records into the KAPipe question format
    for qa in qas:
        answer_texts: list[str] = []
        for answer_text in qa["answers"]:
            assert isinstance(answer_text, str)
            if answer_text not in answer_texts:
                answer_texts.append(answer_text)

        question = {
            "question_key": f"{output_split}#{qa['qa_id']}",
            "question": qa["question"],
            "timestamp": qa["question_date"],
            "answers": [
                {
                    "answer": answer_text,
                }
                for answer_text in answer_texts
            ],
            "evidence_article_ids": [qa["evidence_doc_id"]],
        }
        questions.append(question)

    return questions


def write_jsonl_records(
    output_file: str,
    records: list[dict[str, Any]],
) -> None:
    # Store normalized intermediate records next to the official raw data
    utils.mkdir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_extraction_module(extraction_file: str) -> ModuleType:
    # Validate and import the official Google DeepMind extraction implementation
    if not os.path.isfile(extraction_file):
        raise FileNotFoundError(
            f"Missing StreamingQA extraction.py: {extraction_file}"
        )

    spec = importlib.util.spec_from_file_location(
        "google_deepmind_streamingqa_extraction",
        extraction_file,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Failed to load StreamingQA extraction.py: {extraction_file}"
        )

    extraction_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extraction_module)
    return extraction_module


def write_docs_and_articles(
    input_dir: str,
    wmt_dir: str,
    extraction_module: ModuleType,
    output_docs_file: str,
    output_articles_file: str,
    required_article_ids: set[str],
) -> dict[str, dict[str, Any]]:
    # Require every document-split English WMT archive used by StreamingQA
    wmt_archive_files = [
        os.path.join(
            wmt_dir,
            f"news-docs.{year}.en.filtered.gz",
        )
        for year in range(2007, 2022)
    ]
    for wmt_archive_file in wmt_archive_files:
        if not os.path.isfile(wmt_archive_file):
            raise FileNotFoundError(
                f"Missing a WMT News Crawl archive: {wmt_archive_file}"
            )

    sorting_keys_file = os.path.join(
        input_dir,
        "wmt_sorting_key_ids.txt.gz",
    )
    if not os.path.isfile(sorting_keys_file):
        raise FileNotFoundError(f"Missing WMT sorting keys: {sorting_keys_file}")

    # Create output directories before streaming the large WMT collection
    for output_file in [
        output_docs_file,
        output_articles_file,
    ]:
        utils.mkdir(os.path.dirname(output_file))

    evidence_articles: dict[str, dict[str, Any]] = {}
    article_count = 0

    docs = extraction_module.get_deduplicated_wmt_docs(
        wmt_archive_files=wmt_archive_files,
        deduplicated_sorting_keys_file=sorting_keys_file,
    )

    # Write normalized docs and the complete article collection in one pass
    with (
        open(output_docs_file, "w", encoding="utf-8") as docs_file,
        open(output_articles_file, "w", encoding="utf-8") as articles_file,
    ):
        for doc in tqdm(
            docs,
            total=11_393_471,
            desc="Writing StreamingQA docs and articles",
        ):
            timestamp = timestamp_to_date(doc.publication_ts)
            text = normalize_text(doc.text.decode("utf-8", errors="replace"))
            metadata = {
                "dataset": "StreamingQA",
                "source": "WMT News Crawl",
                "source_version": "document-split",
                "timestamp_type": "publication_ts",
            }
            normalized_doc = {
                "doc_id": doc.sorting_key,
                "text": text,
                "timestamp": doc.publication_ts,
                "date": timestamp,
                "metadata": metadata,
            }
            article = {
                "article_id": doc.sorting_key,
                "doc_id": doc.sorting_key,
                "title": doc.sorting_key,
                "text": text,
                "timestamp": timestamp,
                "source": "WMT News Crawl",
                "url": None,
                "triples": [],
                "metadata": metadata,
            }
            docs_file.write(
                json.dumps(normalized_doc, ensure_ascii=False) + "\n"
            )
            articles_file.write(
                json.dumps(article, ensure_ascii=False) + "\n"
            )
            article_count += 1

            # Retain all evidence articles needed to create gold contexts
            if doc.sorting_key in required_article_ids:
                evidence_articles[doc.sorting_key] = article

    print(
        f"Processed and saved {article_count} normalized docs into "
        f"{output_docs_file}"
    )
    print(
        f"Processed and saved {article_count} articles into "
        f"{output_articles_file}"
    )
    return evidence_articles


def build_gold_contexts(
    questions: list[dict[str, Any]],
    evidence_articles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gold_contexts: list[dict[str, Any]] = []

    # Resolve each official evidence ID into its complete WMT article
    for question in questions:
        contexts = [
            evidence_articles[evidence_article_id]
            for evidence_article_id in question["evidence_article_ids"]
        ]
        gold_contexts.append(
            {
                "question_key": question["question_key"],
                "contexts": contexts,
            }
        )

    return gold_contexts


def read_gzip_jsonl(input_file: str) -> Iterator[dict[str, Any]]:
    # Stream compressed JSONL records to avoid unnecessary copies
    with gzip.open(input_file, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def timestamp_to_date(timestamp: int) -> str:
    # Convert Unix seconds into a UTC calendar date
    return datetime.datetime.fromtimestamp(
        timestamp,
        tz=datetime.timezone.utc,
    ).date().isoformat()


def normalize_text(text: str) -> str:
    # Repair only C1 characters that correspond to Windows-1252 punctuation
    translation_table: dict[int, str] = {
        0x0080: "€",
        0x0082: "‚",
        0x0083: "ƒ",
        0x0084: "„",
        0x0085: "…",
        0x0086: "†",
        0x0087: "‡",
        0x0088: "ˆ",
        0x0089: "‰",
        0x008A: "Š",
        0x008B: "‹",
        0x008C: "Œ",
        0x008E: "Ž",
        0x0091: "‘",
        0x0092: "’",
        0x0093: "“",
        0x0094: "”",
        0x0095: "•",
        0x0096: "–",
        0x0097: "—",
        0x0098: "˜",
        0x0099: "™",
        0x009A: "š",
        0x009B: "›",
        0x009C: "œ",
        0x009E: "ž",
        0x009F: "Ÿ",
    }
    return text.translate(translation_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wmt_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--extraction_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_questions_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_articles_file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
