#!/usr/bin/env bash

set -euo pipefail


ERASE=/home/nishida/storage/dataset/CLARK-News/ERASE
ERASE_COMMIT=b93f898780b3cf40fedd4e9a00c51af9a27688bc

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download the official repository only when it is not already available
# Results:
#   - ERASE/
if [ ! -d "${ERASE}/.git" ]; then
    mkdir -p "$(dirname "${ERASE}")"
    git clone https://github.com/belindal/ERASE "${ERASE}"
    git -C "${ERASE}" switch --detach "${ERASE_COMMIT}"
fi

# Require the tested dataset revision without modifying an existing clone
CURRENT_ERASE_COMMIT=$(git -C "${ERASE}" rev-parse HEAD)
if [ "${CURRENT_ERASE_COMMIT}" != "${ERASE_COMMIT}" ]; then
    echo "Unexpected ERASE revision: ${CURRENT_ERASE_COMMIT}" >&2
    echo "Expected revision: ${ERASE_COMMIT}" >&2
    exit 1
fi

# Results:
#   - STORAGE_DATA/articles/clark_news/articles.jsonl
#   - STORAGE_DATA/qa/clark_news/questions.json
#   - STORAGE_DATA/qa/clark_news/questions.gold_contexts.json
python prepare_clark_news.py \
    --input_dir "${ERASE}/CLARK_news/full" \
    --output_articles_file "${STORAGE_DATA}/articles/clark_news/articles.jsonl" \
    --output_questions_file "${STORAGE_DATA}/qa/clark_news/questions.json" \
    --output_gold_contexts_file "${STORAGE_DATA}/qa/clark_news/questions.gold_contexts.json"

# Results:
#   - STORAGE_DATA/qa/clark_news/questions_filtered.json
python filter_clark_news.py \
    --input_questions_file "${STORAGE_DATA}/qa/clark_news/questions.json" \
    --output_questions_file "${STORAGE_DATA}/qa/clark_news/questions_filtered.json" \
    --random_seed 0
