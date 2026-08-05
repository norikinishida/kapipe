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

    # Convert the official training and validation splits
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
    # Stream the HotpotQA distractor configuration from Hugging Face
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split=source_split,
        streaming=True,
    )
    assert isinstance(dataset, IterableDataset)

    questions: list[dict[str, Any]] = []
    gold_contexts: list[dict[str, Any]] = []
    sentence_level_gold_contexts: list[dict[str, Any]] = []
    gold_contexts_with_distractors: list[dict[str, Any]] = []

    for data in tqdm(dataset, desc=f"Processing {source_split}"):
        # Validate the primary QA fields.
        assert isinstance(data["id"], str)
        assert isinstance(data["question"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["type"], str)
        assert isinstance(data["level"], str)

        question_key = f"hotpotqa-{output_split}-{data['id']}"

        question = {
            "question_key": question_key,
            "question": data["question"],
            "answers": [
                {
                    "answer": data["answer"],
                }
            ],
            "type": data["type"],
            "level": data["level"],
        }
        questions.append(question)

        # Validate the provided context fields
        original_contexts = data["context"]
        assert isinstance(original_contexts, dict)

        context_titles: list[str] = original_contexts["title"]
        context_sentence_lists: list[list[str]] = original_contexts["sentences"]

        assert isinstance(context_titles, list)
        assert isinstance(context_sentence_lists, list)
        assert len(context_titles) == len(context_sentence_lists)

        # Convert all provided paragraphs into the common context format
        all_passages: list[dict[str, str]] = []
        sentences_by_title: dict[str, list[str]] = {}

        for title, sentences in zip(
            context_titles,
            context_sentence_lists,
            strict=True,
        ):
            assert isinstance(title, str)
            assert isinstance(sentences, list)
            assert all(isinstance(sentence, str) for sentence in sentences)
            assert title not in sentences_by_title

            sentences_by_title[title] = sentences

            passage = {
                "title": title,
                "text": " ".join(s.strip() for s in sentences).strip(),
            }
            all_passages.append(passage)

        # Validate the sentence-level supporting-fact annotations
        original_supporting_facts: dict[str, Any] = data["supporting_facts"]
        assert isinstance(original_supporting_facts, dict)

        supporting_titles: list[str] = original_supporting_facts["title"]
        supporting_sentence_indices: list[int] = original_supporting_facts["sent_id"]
        assert isinstance(supporting_titles, list)
        assert isinstance(supporting_sentence_indices, list)
        assert len(supporting_titles) == len(supporting_sentence_indices)

        # Resolve supporting facts against the provided paragraphs
        sentence_level_passages: list[dict[str, str]] = []

        for title, sentence_index in zip(
            supporting_titles,
            supporting_sentence_indices,
            strict=True,
        ):
            assert isinstance(title, str)
            assert isinstance(sentence_index, int)
            assert title in sentences_by_title
            # assert 0 <= sentence_index < len(sentences_by_title[title])
            # Skip malformed supporting-fact annotations
            if not 0 <= sentence_index < len(sentences_by_title[title]):
                print(
                    "Skipping an invalid supporting fact: "
                    f"id={data['id']}, "
                    f"title={title}, "
                    f"sent_id={sentence_index}, "
                    f"sentence_count={len(sentences_by_title[title])}"
                )
                continue

            sentence_level_passage = {
                "title": title,
                "text": sentences_by_title[title][sentence_index].strip(),
            }
            sentence_level_passages.append(
                sentence_level_passage
            )

        # Select complete paragraphs containing supporting facts
        supporting_title_set = set(supporting_titles)

        gold_passages = [
            p 
            for p in all_passages
            if p["title"] in supporting_title_set
        ]

        # Verify that every supporting title has a corresponding paragraph
        assert {
            p["title"]
            for p in gold_passages
        } == supporting_title_set

        gold_contexts.append(
            {
                "question_key": question_key,
                "contexts": gold_passages,
            }
        )

        sentence_level_gold_contexts.append(
            {
                "question_key": question_key,
                "contexts": sentence_level_passages,
            }
        )

        gold_contexts_with_distractors.append(
            {
                "question_key": question_key,
                "contexts": all_passages,
            }
        )

    # Define the four output files
    output_file_path = os.path.join(
        output_dir,
        f"{output_split}.json",
    )
    gold_contexts_output_file_path = os.path.join(
        output_dir,
        f"{output_split}.gold_contexts.json",
    )
    sentence_level_gold_contexts_output_file_path = os.path.join(
        output_dir,
        f"{output_split}.gold_contexts_at_sentence_level.json",
    )
    gold_contexts_with_distractors_output_file_path = os.path.join(
        output_dir,
        f"{output_split}.gold_contexts_with_distractors.json",
    )

    # Save the QA instances and the three context variants
    utils.write_json(
        output_file_path,
        questions,
    )
    utils.write_json(
        gold_contexts_output_file_path,
        gold_contexts,
    )
    utils.write_json(
        sentence_level_gold_contexts_output_file_path,
        sentence_level_gold_contexts,
    )
    utils.write_json(
        gold_contexts_with_distractors_output_file_path,
        gold_contexts_with_distractors,
    )

    print(
        f"Processed and saved {len(questions)} questions "
        f"into {output_file_path}"
    )
    print(
        f"Processed and saved {len(gold_contexts)} "
        f"gold-context instances into "
        f"{gold_contexts_output_file_path}"
    )
    print(
        f"Processed and saved "
        f"{len(sentence_level_gold_contexts)} "
        f"sentence-level gold-context instances into "
        f"{sentence_level_gold_contexts_output_file_path}"
    )
    print(
        f"Processed and saved "
        f"{len(gold_contexts_with_distractors)} "
        f"gold-context-with-distractor instances into "
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