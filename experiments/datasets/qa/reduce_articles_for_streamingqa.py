import argparse
import json
import logging
import os
from typing import Any

from tqdm import tqdm

from kapipe import utils
from kapipe.passage_retrieval import Contriever


logger = logging.getLogger(__name__)


EXPECTED_TEST_QUESTION_COUNT = 1_120
TOP_K = 10
MODEL_NAME = "facebook/contriever-msmarco"
MAX_PASSAGE_LENGTH = 512
POOLING_METHOD = "average"
NORMALIZE = False
METRIC = "inner-product"
INDEXING_BATCH_SIZE = 1024
SEARCH_BATCH_SIZE = 10


def main(args: argparse.Namespace) -> None:
    input_questions_file: str = args.input_questions_file
    input_gold_contexts_file: str = args.input_gold_contexts_file
    input_articles_file: str = args.input_articles_file
    index_dir: str = args.index_dir
    output_articles_file: str = args.output_articles_file

    # Load every filtered test question without applying subsampling
    logger.info("Loading filtered test questions from %s", input_questions_file)
    questions = utils.read_json(input_questions_file)
    if len(questions) != EXPECTED_TEST_QUESTION_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_TEST_QUESTION_COUNT} filtered test questions, "
            f"but found {len(questions)}"
        )

    logger.info("Loaded %d filtered test questions", len(questions))

    # Load the complete test gold contexts and index them by question key
    gold_contexts = utils.read_json(input_gold_contexts_file)
    question_key_to_gold_contexts = {
        gold_contexts_for_question["question_key"]: (
            gold_contexts_for_question["contexts"]
        )
        for gold_contexts_for_question in gold_contexts
    }
    if len(question_key_to_gold_contexts) != len(gold_contexts):
        raise ValueError("Found duplicate question keys in gold contexts")

    # Instantiate Contriever
    logger.info("Loading Contriever model: %s", MODEL_NAME)
    retriever = Contriever(
        model_name=MODEL_NAME,
        max_passage_length=MAX_PASSAGE_LENGTH,
        pooling_method=POOLING_METHOD,
        normalize=NORMALIZE,
        metric=METRIC,
    )

    # Load the complete corpus with byte-level progress reporting
    input_articles_size = os.path.getsize(input_articles_file)
    logger.info(
        "Loading complete article corpus from %s (%d bytes)",
        input_articles_file,
        input_articles_size,
    )
    articles: list[dict[str, Any]] = []
    with open(input_articles_file, "rb") as file:
        with tqdm(
            total=input_articles_size,
            desc="Loading StreamingQA articles",
            unit="B",
            unit_scale=True,
            mininterval=1.0,
        ) as progress:
            for line in file:
                progress.update(len(line))
                if line.strip():
                    articles.append(json.loads(line))
    logger.info("Loaded %d article records", len(articles))

    # Build and save the complete Contriever index
    logger.info(
        "Building Contriever index in %s with batch size %d",
        index_dir,
        INDEXING_BATCH_SIZE,
    )
    retriever.make_index(
        passages=articles,
        index_dir=index_dir,
        batch_size=INDEXING_BATCH_SIZE,
    )
    logger.info("Completed Contriever indexing")

    # Collect every official gold article before adding retrieval results
    selected_article_ids: set[str] = set()
    for question in questions:
        contexts = question_key_to_gold_contexts[question["question_key"]]
        if len(contexts) != 1:
            raise ValueError(
                "Expected one gold context for "
                f"{question['question_key']}, but found {len(contexts)}"
            )
        selected_article_ids.add(contexts[0]["article_id"])
    gold_article_count = len(selected_article_ids)
    logger.info("Collected %d unique gold article IDs", gold_article_count)

    # Retrieve the top-10 articles for every filtered test question
    logger.info(
        "Retrieving top-%d articles for %d questions",
        TOP_K,
        len(questions),
    )
    for start_i in tqdm(
        range(0, len(questions), SEARCH_BATCH_SIZE),
        desc="Retrieving StreamingQA test articles",
    ):
        batch_questions = questions[start_i:start_i + SEARCH_BATCH_SIZE]
        batch_passages = retriever.search(
            queries=[question["question"] for question in batch_questions],
            top_k=TOP_K,
        )
        for passages in batch_passages:
            if len(passages) != TOP_K:
                raise ValueError(
                    f"Expected {TOP_K} retrieved articles, "
                    f"but found {len(passages)}"
                )
            selected_article_ids.update(
                passage["article_id"]
                for passage in passages
            )

    logger.info(
        "Collected %d unique gold and retrieved article IDs",
        len(selected_article_ids),
    )

    # Read only the selected article records and remove repeated article IDs
    selected_articles: dict[str, dict[str, Any]] = {}
    logger.info(
        "Scanning %s to resolve %d selected article IDs",
        input_articles_file,
        len(selected_article_ids),
    )
    with open(input_articles_file, "rb") as file:
        with tqdm(
            total=input_articles_size,
            desc="Collecting selected StreamingQA articles",
            unit="B",
            unit_scale=True,
            mininterval=1.0,
        ) as progress:
            for line in file:
                progress.update(len(line))
                if not line.strip():
                    continue
                article = json.loads(line)
                article_id = article["article_id"]
                if (
                    article_id in selected_article_ids
                    and article_id not in selected_articles
                ):
                    selected_articles[article_id] = article
    logger.info(
        "Resolved %d unique selected articles",
        len(selected_articles),
    )

    # Fail if an index article cannot be resolved in the source article file
    missing_article_ids = selected_article_ids - set(selected_articles)
    if missing_article_ids:
        examples = sorted(missing_article_ids)[:10]
        raise ValueError(f"Missing selected StreamingQA articles: {examples}")

    # Write exactly one record per article in article-ID order
    logger.info(
        "Writing %d articles in article-ID order to %s",
        len(selected_articles),
        output_articles_file,
    )
    output_dir = os.path.dirname(output_articles_file)
    if output_dir:
        utils.mkdir(output_dir)
    with open(output_articles_file, "w", encoding="utf-8") as file:
        for article_id in sorted(selected_articles):
            article = selected_articles[article_id]
            file.write(json.dumps(article, ensure_ascii=False) + "\n")

    logger.info("Gold articles: %d", gold_article_count)
    logger.info(
        "Gold and retrieved articles: %d",
        len(selected_articles),
    )
    logger.info(
        "Saved filtered test articles to %s",
        output_articles_file,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_questions_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_gold_contexts_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_articles_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--index_dir",
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
