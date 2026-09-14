#!/usr/bin/env bash

set -euo pipefail


FANOUTQA=/home/nishida/storage/dataset/FanOutQA
FANOUTQA_COMMIT=989f4c40d9deea1ecb0897d7a17a9c0fe20d5c33

WIKIPEDIA_DUMP_BASE=enwiki-20231120
WIKIPEDIA_DUMP_URL=https://datasets.mechanus.zhu.codes/fanoutqa/${WIKIPEDIA_DUMP_BASE}-pages-articles-multistream.xml.bz2
WIKIPEDIA_DUMP_FILENAME=${WIKIPEDIA_DUMP_BASE}-pages-articles-multistream.xml.bz2

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets
OUTPUT_ARTICLES_FILE="${STORAGE_DATA}/fanoutqa/corpus/${WIKIPEDIA_DUMP_BASE}.articles.jsonl"
OUTPUT_QA_DIR="${STORAGE_DATA}/fanoutqa/qa"


# Download the official repository only when it is not already available
# Results:
#   - FANOUTQA/fanoutqa/
if [ ! -d "${FANOUTQA}/fanoutqa/.git" ]; then
    mkdir -p "${FANOUTQA}"
    git clone \
        https://github.com/zhudotexe/fanoutqa.git \
        "${FANOUTQA}/fanoutqa"
    git -C "${FANOUTQA}/fanoutqa" \
        switch --detach "${FANOUTQA_COMMIT}"
fi

# Validate that the current FanOutQA repository is at the expected commit
CURRENT_FANOUTQA_COMMIT=$(
    git -C "${FANOUTQA}/fanoutqa" rev-parse HEAD
)
if [ "${CURRENT_FANOUTQA_COMMIT}" != "${FANOUTQA_COMMIT}" ]; then
    echo "Unexpected FanOutQA revision: ${CURRENT_FANOUTQA_COMMIT}" >&2
    echo "Expected revision: ${FANOUTQA_COMMIT}" >&2
    exit 1
fi

# Download the official Wikipedia snapshot aligned with the November 2023 data
# Results:
#   - FANOUTQA/wikipedia/FANOUTQA_DUMP_FILENAME
mkdir -p "${FANOUTQA}/wikipedia"
wget -c \
    "${WIKIPEDIA_DUMP_URL}" \
    -P "${FANOUTQA}/wikipedia/"

# Extract the complete Wikipedia snapshot when it is not already available
# Results:
#   - FANOUTQA/wikipedia/FANOUTQA_DUMP_FILENAME.extracted/
if [ ! -d "${FANOUTQA}/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted" ]; then
    python -m wikiextractor.WikiExtractor \
        "${FANOUTQA}/wikipedia/${WIKIPEDIA_DUMP_FILENAME}" \
        -o "${FANOUTQA}/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted" \
        --json \
        --processes 4 \
        -b 1G
fi

# Results:
#   - STORAGE_DATA/fanoutqa/corpus/WIKIPEDIA_DUMP_BASE.articles.jsonl
if [ ! -f "${STORAGE_DATA}/fanoutqa/corpus/${WIKIPEDIA_DUMP_BASE}.articles.jsonl" ]; then
    python prepare_fanoutqa_corpus.py \
        --input_dir "${FANOUTQA}/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted" \
        --output_file "${OUTPUT_ARTICLES_FILE}"
fi

# Results:
#   - STORAGE_DATA/fanoutqa/qa/{dev,test}.json
#   - STORAGE_DATA/fanoutqa/qa/{dev,test}.gold_contexts.json
python prepare_fanoutqa_qa.py \
    --input_dev_file "${FANOUTQA}/fanoutqa/fanoutqa/data/fanout-final-dev-nov23.json" \
    --input_test_file "${FANOUTQA}/fanoutqa/fanoutqa/data/fanout-final-test-nov23.json" \
    --input_articles_file "${OUTPUT_ARTICLES_FILE}" \
    --output_dir "${OUTPUT_QA_DIR}"
