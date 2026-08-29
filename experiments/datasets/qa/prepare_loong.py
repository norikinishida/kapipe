import argparse
import glob
import json
import os
import re
from typing import Any

from tqdm import tqdm

from kapipe import utils


DOMAINS: list[str] = [
    "financial",
    "legal",
    "paper",
]

DATASET_NAMES: list[str] = [
    "financial_en",
    "financial_zh",
    "legal_zh",
    "paper_en",
]

TASK_LEVEL_TO_TYPE: dict[int, str] = {
    1: "Spotlight Locating",
    2: "Comparison",
    3: "Clustering",
    4: "Chain of Reasoning",
}

LEGAL_CONTENT_ONLY_INSTRUCTION = (
    "阅读以上判决文书，我将给你若干份判决结果："
)


def main(args: argparse.Namespace) -> None:
    input_questions_file: str = args.input_questions_file
    input_documents_dir: str = args.input_documents_dir
    output_articles_dir: str = args.output_articles_dir
    output_qa_dir: str = args.output_qa_dir

    # Require every released Loong input before starting conversion
    if not os.path.isfile(input_questions_file):
        raise FileNotFoundError(
            f"Missing the Loong question file: {input_questions_file}"
        )
    for domain in DOMAINS:
        domain_dir = os.path.join(input_documents_dir, domain)
        if not os.path.isdir(domain_dir):
            raise FileNotFoundError(
                f"Missing the Loong document directory: {domain_dir}"
            )

    # Load the released annotations and the shared legal-document mapping
    original_questions = load_original_questions(
        input_questions_file=input_questions_file,
    )
    legal_documents = load_legal_documents(
        input_documents_dir=input_documents_dir,
    )

    # Convert all questions into separate domain-language datasets
    questions_by_dataset, gold_contexts_by_dataset, articles_by_dataset = (
        convert_dataset(
            original_questions=original_questions,
            input_documents_dir=input_documents_dir,
            legal_documents=legal_documents,
        )
    )

    # Create only the established dataset-level output directories
    utils.mkdir(output_articles_dir)
    utils.mkdir(output_qa_dir)

    # Save flat domain-language files without adding subdirectories
    for dataset_name in DATASET_NAMES:
        output_articles_file = os.path.join(
            output_articles_dir,
            f"{dataset_name}.articles.jsonl",
        )
        output_questions_file = os.path.join(
            output_qa_dir,
            f"{dataset_name}.json",
        )
        output_gold_contexts_file = os.path.join(
            output_qa_dir,
            f"{dataset_name}.gold_contexts.json",
        )

        utils.write_jsonl(
            output_articles_file,
            articles_by_dataset[dataset_name],
            ensure_ascii=False,
        )
        utils.write_json(
            output_questions_file,
            questions_by_dataset[dataset_name],
            ensure_ascii=False,
        )
        utils.write_json(
            output_gold_contexts_file,
            gold_contexts_by_dataset[dataset_name],
            ensure_ascii=False,
        )

        print(
            f"Processed and saved {len(articles_by_dataset[dataset_name])} "
            f"{dataset_name} articles into {output_articles_file}"
        )
        print(
            f"Processed and saved {len(questions_by_dataset[dataset_name])} "
            f"{dataset_name} questions into {output_questions_file}"
        )
        print(
            f"Processed and saved "
            f"{len(gold_contexts_by_dataset[dataset_name])} {dataset_name} "
            f"gold-context instances into {output_gold_contexts_file}"
        )


def load_original_questions(
    input_questions_file: str,
) -> list[dict[str, Any]]:
    original_questions: list[dict[str, Any]] = []

    # Read the official newline-delimited annotations without reordering them
    with open(input_questions_file, encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            if not line.strip():
                raise ValueError(
                    f"Empty line in {input_questions_file}: {line_index + 1}"
                )
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError(
                    f"Expected an object on line {line_index + 1}: "
                    f"{type(data).__name__}"
                )
            original_questions.append(data)

    return original_questions


def load_legal_documents(
    input_documents_dir: str,
) -> dict[str, dict[str, str]]:
    input_legal_file = os.path.join(
        input_documents_dir,
        "legal",
        "legal.json",
    )
    if not os.path.isfile(input_legal_file):
        raise FileNotFoundError(
            f"Missing the Loong legal-document file: {input_legal_file}"
        )

    # Load the one released mapping from document names to document sections
    legal_documents = utils.read_json(input_legal_file)
    if not isinstance(legal_documents, dict):
        raise TypeError(
            f"Expected a legal-document object: {input_legal_file}"
        )

    for document_id, document in legal_documents.items():
        if not isinstance(document_id, str):
            raise TypeError("Expected every legal document ID to be a string")
        if not isinstance(document, dict):
            raise TypeError(
                f"Expected a legal document object: {document_id}"
            )
        # Require only the fields that the official prompt presents to models
        required_document_fields = {"content", "result"}
        if not required_document_fields.issubset(document):
            raise ValueError(
                f"Missing legal document fields for {document_id}: "
                f"{sorted(document)}"
            )
        if not isinstance(document["content"], str):
            raise TypeError(f"Expected string content for {document_id}")
        if not isinstance(document["result"], str):
            raise TypeError(f"Expected string result for {document_id}")

    return legal_documents


def convert_dataset(
    original_questions: list[dict[str, Any]],
    input_documents_dir: str,
    legal_documents: dict[str, dict[str, str]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, str]]],
]:
    questions_by_dataset: dict[str, list[dict[str, Any]]] = {
        dataset_name: []
        for dataset_name in DATASET_NAMES
    }
    gold_contexts_by_dataset: dict[str, list[dict[str, Any]]] = {
        dataset_name: []
        for dataset_name in DATASET_NAMES
    }
    articles_by_dataset: dict[str, list[dict[str, str]]] = {
        dataset_name: []
        for dataset_name in DATASET_NAMES
    }
    document_id_to_article_by_dataset: dict[
        str,
        dict[str, dict[str, str]],
    ] = {
        dataset_name: {}
        for dataset_name in DATASET_NAMES
    }
    seen_question_keys: set[str] = set()

    # Convert each official test instance exactly once
    for data in tqdm(original_questions, desc="Converting Loong questions"):
        validate_original_question(data=data)

        question_key = data["id"]
        if question_key in seen_question_keys:
            raise ValueError(f"Duplicate Loong question key: {question_key}")
        seen_question_keys.add(question_key)

        domain = data["type"]
        dataset_name = f"{domain}_{data['language']}"
        if dataset_name not in DATASET_NAMES:
            raise ValueError(f"Unexpected Loong dataset: {dataset_name}")
        question = build_question(data=data)
        questions_by_dataset[dataset_name].append(question)

        # Expand each released document search key into independent Passages
        contexts: list[dict[str, str]] = []
        for source_document_key in data["doc"]:
            resolved_documents = resolve_documents(
                data=data,
                source_document_key=source_document_key,
                input_documents_dir=input_documents_dir,
                legal_documents=legal_documents,
            )
            for document_id, text in resolved_documents:
                article = {
                    "text": text,
                    "document_id": document_id,
                }
                contexts.append(article)

                # Require one immutable Passage for each exact document ID
                document_id_to_article = document_id_to_article_by_dataset[
                    dataset_name
                ]
                if document_id in document_id_to_article:
                    if article != document_id_to_article[document_id]:
                        raise ValueError(
                            f"Conflicting text for {domain} document: "
                            f"{document_id}"
                        )
                else:
                    document_id_to_article[document_id] = article
                    articles_by_dataset[dataset_name].append(article)

        gold_contexts_by_dataset[dataset_name].append(
            {
                "question_key": question_key,
                "contexts": contexts,
            }
        )

    return (
        questions_by_dataset,
        gold_contexts_by_dataset,
        articles_by_dataset,
    )


def validate_original_question(data: dict[str, Any]) -> None:
    required_fields = {
        "level",
        "set",
        "length",
        "type",
        "language",
        "question",
        "instruction",
        "prompt_template",
        "doc",
        "answer",
        "shuffle_doc",
        "id",
    }
    if set(data) != required_fields:
        raise ValueError(
            f"Unexpected Loong fields for {data.get('id')}: "
            f"{sorted(data)}"
        )

    # Validate every released field before intentionally discarding some fields
    if data["level"] not in {1, 2, 3, 4}:
        raise ValueError(f"Unexpected task level: {data['level']}")
    if data["set"] not in {1, 2, 3, 4}:
        raise ValueError(f"Unexpected length set: {data['set']}")
    if not isinstance(data["length"], int):
        raise TypeError(f"Expected integer length for {data['id']}")
    if data["type"] not in DOMAINS:
        raise ValueError(f"Unexpected Loong domain: {data['type']}")
    if data["language"] not in {"en", "zh"}:
        raise ValueError(f"Unexpected Loong language: {data['language']}")
    if not isinstance(data["question"], str):
        raise TypeError(f"Expected string question for {data['id']}")
    if not isinstance(data["instruction"], str):
        raise TypeError(f"Expected string instruction for {data['id']}")
    if not isinstance(data["prompt_template"], str):
        raise TypeError(f"Expected string prompt template for {data['id']}")
    if not isinstance(data["doc"], list) or not data["doc"]:
        raise TypeError(f"Expected a non-empty document list for {data['id']}")
    if not all(isinstance(document_id, str) for document_id in data["doc"]):
        raise TypeError(f"Expected string document IDs for {data['id']}")
    if not isinstance(data["answer"], (str, list, dict)):
        raise TypeError(f"Unexpected answer type for {data['id']}")
    if not isinstance(data["shuffle_doc"], bool):
        raise TypeError(f"Expected boolean shuffle_doc for {data['id']}")
    if not isinstance(data["id"], str):
        raise TypeError("Expected every Loong ID to be a string")


def build_question(data: dict[str, Any]) -> dict[str, Any]:
    original_question = data["question"]
    original_instruction = data["instruction"]

    # Combine the original fields using one template-guided method
    question = combine_question_fields(
        original_question=original_question,
        original_instruction=original_instruction,
        prompt_template=data["prompt_template"],
    )

    return {
        "question_key": data["id"],
        "original_question": original_question,
        "original_instruction": original_instruction,
        "question": question,
        "answers": convert_answer(answer=data["answer"]),
        "task_level": str(data["level"]),
        "task_type": TASK_LEVEL_TO_TYPE[data["level"]],
        "domain": data["type"],
        "language": data["language"],
    }


def combine_question_fields(
    original_question: str,
    original_instruction: str,
    prompt_template: str,
) -> str:
    # Remove each complete document block, including its domain-specific heading
    template_blocks = re.split(r"\n[ \t]*\n", prompt_template.strip())
    retained_blocks = [
        block
        for block in template_blocks
        if "{docs}" not in block
    ]
    question_template = "\n\n".join(retained_blocks)

    # Require the template to consume every non-empty source field
    if "{instruction}" not in question_template:
        raise ValueError("The prompt template does not contain {instruction}")
    if original_question and "{question}" not in question_template:
        raise ValueError(
            "The prompt template does not contain the non-empty {question}"
        )

    # Substitute only the two released fields and preserve surrounding wording
    question = question_template.replace(
        "{instruction}",
        original_instruction,
    ).replace(
        "{question}",
        original_question,
    ).strip()

    if "{docs}" in question or "{instruction}" in question or "{question}" in question:
        raise ValueError("Unresolved placeholder in the combined Loong question")
    if not question:
        raise ValueError("Combined Loong question must not be empty")
    return question


def convert_answer(answer: str | list[Any] | dict[str, Any]) -> list[dict[str, str]]:
    # Preserve ordinary string answers without adding an answer type
    if isinstance(answer, str):
        return [
            {
                "answer": answer,
            }
        ]

    # Preserve one complete structured answer as a compact JSON string
    return [
        {
            "answer": json.dumps(
                answer,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "answer_type": "json",
        }
    ]


def resolve_documents(
    data: dict[str, Any],
    source_document_key: str,
    input_documents_dir: str,
    legal_documents: dict[str, dict[str, str]],
) -> list[tuple[str, str]]:
    domain = data["type"]

    # Expand one financial search key into complete, separately identified files
    if domain == "financial":
        return resolve_financial_documents(
            source_document_key=source_document_key,
            task_level=data["level"],
            input_documents_dir=input_documents_dir,
        )

    # Read one exact academic-paper Markdown file without rewriting it
    if domain == "paper":
        input_file = os.path.join(
            input_documents_dir,
            "paper",
            source_document_key,
        )
        if not os.path.isfile(input_file):
            raise FileNotFoundError(
                f"Missing Loong paper document: {input_file}"
            )
        with open(input_file, encoding="utf-8") as file:
            return [(source_document_key, file.read())]

    # Resolve one legal.json entry while retaining an explicit section boundary
    if source_document_key not in legal_documents:
        raise KeyError(
            f"Missing Loong legal document: {source_document_key}"
        )
    legal_document = legal_documents[source_document_key]
    if (
        data["level"] == 4
        and LEGAL_CONTENT_ONLY_INSTRUCTION in data["instruction"]
    ):
        text = legal_document["content"]
    else:
        text = legal_document["content"] + "\n\n" + legal_document["result"]
    return [(source_document_key, text)]


def resolve_financial_documents(
    source_document_key: str,
    task_level: int,
    input_documents_dir: str,
) -> list[tuple[str, str]]:
    # Treat the released search key literally even when it contains glob characters
    escaped_document_key = glob.escape(source_document_key)
    if task_level == 4:
        filename_pattern = f"*{escaped_document_key}*.txt"
    else:
        filename_pattern = f"*2024-{escaped_document_key}*.txt"
    input_pattern = os.path.join(
        input_documents_dir,
        "financial",
        filename_pattern,
    )
    matched_files = sorted(glob.glob(input_pattern))

    # Require the released search key to match at least one source file
    if not matched_files:
        raise FileNotFoundError(
            f"Missing financial documents for {source_document_key}"
        )

    # Preserve every matched file as one Passage with its exact filename as ID
    resolved_documents: list[tuple[str, str]] = []
    for input_file in matched_files:
        document_id = os.path.basename(input_file)
        with open(input_file, encoding="utf-8") as file:
            text = file.read()
        resolved_documents.append((document_id, text))
    return resolved_documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_questions_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_documents_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_articles_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_qa_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
