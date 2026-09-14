#!/usr/bin/env bash

set -euo pipefail


FRAMES=/home/nishida/storage/dataset/FRAMES
FRAMES_REVISION=58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef
FRAMES_TEST_URL="https://huggingface.co/datasets/google/frames-benchmark/resolve/${FRAMES_REVISION}/test.tsv"

WIKIPEDIA_DATE=20230601

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets
OUTPUT_ARTICLES_FILE="${STORAGE_DATA}/frames/corpus/enwiki-${WIKIPEDIA_DATE}.articles.jsonl"
OUTPUT_QA_DIR="${STORAGE_DATA}/frames/qa"


# Download the official FRAMES test set from a fixed dataset revision
# Results:
#   - FRAMES/test.tsv
mkdir -p "${FRAMES}"
wget -c \
    "${FRAMES_TEST_URL}" \
    -O "${FRAMES}/test.tsv"

# Results:
#   - STORAGE_DATA/frames/corpus/enwiki-WIKIPEDIA_DATE.articles.jsonl
#   - STORAGE_DATA/frames/qa/test.json
#   - STORAGE_DATA/frames/qa/test.gold_contexts.json
python prepare_frames_qa_and_corpus.py \
    --input_questions_file "${FRAMES}/test.tsv" \
    --wikipedia_config "${WIKIPEDIA_DATE}.en" \
    --output_articles_file "${OUTPUT_ARTICLES_FILE}" \
    --output_questions_file "${OUTPUT_QA_DIR}/test.json" \
    --output_gold_contexts_file "${OUTPUT_QA_DIR}/test.gold_contexts.json"
