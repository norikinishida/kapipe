import argparse
import os
from typing import Any

from datasets import IterableDataset, load_dataset
from tqdm import tqdm

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    output_articles_file: str = args.output_articles_file
    output_dir: str = args.output_dir

    # Create the output directory
    utils.mkdir(output_dir)

    # Stream the only official LongBench v2 split from Hugging Face
    dataset = load_dataset(
        "THUDM/LongBench-v2",
        split="train",
        streaming=True,
    )
    assert isinstance(dataset, IterableDataset)

    questions: list[dict[str, Any]] = []
    gold_contexts: list[dict[str, Any]] = []
    articles: list[dict[str, str]] = []
    seen_article_texts: set[str] = set()

    # Convert every official instance without changing its question or context
    for data in tqdm(dataset, desc="Processing train"):
        # Validate every field defined by the official data format
        assert isinstance(data["_id"], str)
        assert isinstance(data["domain"], str)
        assert isinstance(data["sub_domain"], str)
        assert data["difficulty"] in ["easy", "hard"]
        assert data["length"] in ["short", "medium", "long"]
        assert isinstance(data["question"], str)
        assert isinstance(data["choice_A"], str)
        assert isinstance(data["choice_B"], str)
        assert isinstance(data["choice_C"], str)
        assert isinstance(data["choice_D"], str)
        assert data["answer"] in ["A", "B", "C", "D"]
        assert isinstance(data["context"], str)

        # Keep exactly one retrieval article for each distinct context text
        if data["context"] not in seen_article_texts:
            seen_article_texts.add(data["context"])
            articles.append(
                {
                    "text": data["context"],
                }
            )

        # Retain the official split name in the repository-wide question key
        question_key = f"longbench-v2-train-{data['_id']}"

        # Preserve option labels so the answer remains the official letter
        questions.append(
            {
                "question_key": question_key,
                "question": data["question"],
                "candidate_answers": [
                    f"(A) {data['choice_A']}",
                    f"(B) {data['choice_B']}",
                    f"(C) {data['choice_C']}",
                    f"(D) {data['choice_D']}",
                ],
                "answers": [
                    {
                        "answer": data["answer"],
                    }
                ],
                "domain": data["domain"],
                "sub_domain": data["sub_domain"],
                "difficulty": data["difficulty"],
                "length": data["length"],
            }
        )

        # Keep the complete official context as one passage
        gold_contexts.append(
            {
                "question_key": question_key,
                "contexts": [
                    {
                        "text": data["context"],
                    }
                ],
            }
        )

    # Define QA output files using the official split name
    output_questions_file = os.path.join(output_dir, "train.json")
    output_gold_contexts_file = os.path.join(
        output_dir,
        "train.gold_contexts.json",
    )

    # Create the retrieval-corpus directory after validating all records
    utils.mkdir(os.path.dirname(output_articles_file))

    # Save unique contexts as newline-delimited retrieval articles
    utils.write_jsonl(
        output_articles_file,
        articles,
        ensure_ascii=False,
    )

    # Save questions and their corresponding complete contexts separately
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
        f"Processed and saved {len(articles)} unique articles into "
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
