import argparse
import os
from typing import Any

from datasets import IterableDataset, load_dataset
from tqdm import tqdm

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    output_dir: str = args.output_dir

    # Create the output directory
    utils.mkdir(output_dir)

    # Map the official development file to the repository's dev convention
    split_mapping = {
        "train": "train",
        "validation": "dev",
    }

    # Process each split and save the results to the output directory
    for source_split, output_split in split_mapping.items():
        process_split(
            source_split=source_split,
            output_split=output_split,
            output_dir=output_dir,
        )


def process_split(
    source_split: str,
    output_split: str,
    output_dir: str,
) -> None:
    # Select the complete official MuSiQue-Answerable train and dev files
    data_files = {
        "train": "musique_ans_v1.0_train.jsonl",
        "validation": "musique_ans_v1.0_dev.jsonl",
    }

    # Stream the selected MuSiQue split from Hugging Face
    dataset = load_dataset(
        "dgslibisey/MuSiQue",
        data_files=data_files,
        split=source_split,
        streaming=True,
    )
    assert isinstance(dataset, IterableDataset)

    questions: list[dict[str, Any]] = []
    gold_contexts: list[dict[str, Any]] = []
    gold_contexts_with_distractors: list[dict[str, Any]] = []

    # Convert every QA instance and its paragraph annotations
    for data in tqdm(dataset, desc=f"Processing {source_split}"):
        # Validate the primary QA fields
        assert isinstance(data["id"], str)
        assert isinstance(data["question"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["answer_aliases"], list)
        assert all(isinstance(alias, str) for alias in data["answer_aliases"])
        assert data["answerable"] is True

        # Preserve all unique official answer aliases in their original order
        answer_texts: list[str] = []
        for answer_text in [data["answer"], *data["answer_aliases"]]:
            if answer_text not in answer_texts:
                answer_texts.append(answer_text)

        answers = [
            {
                "answer": answer_text,
            }
            for answer_text in answer_texts
        ]

        question_key = f"musique-{output_split}-{data['id']}"

        # Validate and convert every provided paragraph
        original_paragraphs = data["paragraphs"]
        assert isinstance(original_paragraphs, list)

        all_passages: list[dict[str, str]] = []
        gold_passages: list[dict[str, str]] = []
        passage_by_paragraph_idx: dict[int, dict[str, str]] = {}
        paragraph_indices: set[int] = set()
        annotated_supporting_indices: set[int] = set()

        for paragraph in original_paragraphs:
            assert isinstance(paragraph, dict)
            assert isinstance(paragraph["idx"], int)
            assert isinstance(paragraph["title"], str)
            assert isinstance(paragraph["paragraph_text"], str)
            assert isinstance(paragraph["is_supporting"], bool)
            assert paragraph["idx"] not in paragraph_indices

            paragraph_indices.add(paragraph["idx"])

            passage = {
                "title": paragraph["title"],
                "text": paragraph["paragraph_text"].strip(),
            }
            all_passages.append(passage)
            passage_by_paragraph_idx[paragraph["idx"]] = passage

            # Select the paragraphs marked as supporting by MuSiQue
            if paragraph["is_supporting"]:
                annotated_supporting_indices.add(paragraph["idx"])
                gold_passages.append(passage)

        # Replace every decomposition answer and paragraph index with QA fields
        original_question_decomposition = data["question_decomposition"]
        assert isinstance(original_question_decomposition, list)

        question_decomposition: list[dict[str, Any]] = []
        supporting_paragraph_indices: set[int] = set()

        for decomposition_step in original_question_decomposition:
            assert isinstance(decomposition_step, dict)
            assert isinstance(decomposition_step["id"], int)
            assert isinstance(decomposition_step["question"], str)
            assert isinstance(decomposition_step["answer"], str)
            assert isinstance(decomposition_step["paragraph_support_idx"], int)

            paragraph_support_idx = decomposition_step["paragraph_support_idx"]
            assert paragraph_support_idx in passage_by_paragraph_idx
            supporting_paragraph_indices.add(paragraph_support_idx)

            supporting_passage = passage_by_paragraph_idx[paragraph_support_idx]
            question_decomposition.append(
                {
                    "id": decomposition_step["id"],
                    "question": decomposition_step["question"],
                    "answers": [
                        {
                            "answer": decomposition_step["answer"],
                        }
                    ],
                    "contexts": [supporting_passage],
                }
            )

        # Verify that decomposition links and paragraph labels agree exactly
        assert supporting_paragraph_indices == annotated_supporting_indices

        question = {
            "question_key": question_key,
            "question": data["question"],
            "answers": answers,
            "question_decomposition": question_decomposition,
        }
        questions.append(question)

        gold_contexts.append(
            {
                "question_key": question_key,
                "contexts": gold_passages,
            }
        )
        gold_contexts_with_distractors.append(
            {
                "question_key": question_key,
                "contexts": all_passages,
            }
        )

    # Define the three output files
    output_file_path = os.path.join(output_dir, f"{output_split}.json")
    gold_contexts_output_file_path = os.path.join(
        output_dir,
        f"{output_split}.gold_contexts.json",
    )
    gold_contexts_with_distractors_output_file_path = os.path.join(
        output_dir,
        f"{output_split}.gold_contexts_with_distractors.json",
    )

    # Save the QA instances and both paragraph-level context variants
    utils.write_json(
        output_file_path,
        questions,
        ensure_ascii=False,
    )
    utils.write_json(
        gold_contexts_output_file_path,
        gold_contexts,
        ensure_ascii=False,
    )
    utils.write_json(
        gold_contexts_with_distractors_output_file_path,
        gold_contexts_with_distractors,
        ensure_ascii=False,
    )

    print(f"Processed and saved {len(questions)} questions into {output_file_path}")
    print(
        f"Processed and saved {len(gold_contexts)} gold-context instances "
        f"into {gold_contexts_output_file_path}"
    )
    print(
        f"Processed and saved {len(gold_contexts_with_distractors)} "
        "gold-context-with-distractor instances into "
        f"{gold_contexts_with_distractors_output_file_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
