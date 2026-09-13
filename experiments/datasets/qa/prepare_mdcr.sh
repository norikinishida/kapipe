#!/usr/bin/env bash

set -euo pipefail


MDCR=/home/nishida/storage/dataset/MDCR
MDCR_COMMIT=dd9d10b697ab09700b28ec25e03c4931a92ae403

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download the official repository only when it is not already available
# Results:
#   - MDCR/mdcr/
if [ ! -d "${MDCR}/mdcr/.git" ]; then
    mkdir -p "${MDCR}"
    git clone \
        https://github.com/peterbaile/mdcr.git \
        "${MDCR}/mdcr"
    git -C "${MDCR}/mdcr" \
        switch --detach "${MDCR_COMMIT}"
fi

# Validate that the current MDCR repository is at the expected commit
CURRENT_MDCR_COMMIT=$(
    git -C "${MDCR}/mdcr" rev-parse HEAD
)
if [ "${CURRENT_MDCR_COMMIT}" != "${MDCR_COMMIT}" ]; then
    echo "Unexpected MDCR revision: ${CURRENT_MDCR_COMMIT}" >&2
    echo "Expected revision: ${MDCR_COMMIT}" >&2
    exit 1
fi

# Results:
#   - STORAGE_DATA/articles/mdcr/articles.jsonl
#   - STORAGE_DATA/qa/mdcr/questions.json
#   - STORAGE_DATA/qa/mdcr/questions.gold_contexts.json
python prepare_mdcr.py \
    --input_repository_dir "${MDCR}/mdcr" \
    --output_articles_file "${STORAGE_DATA}/articles/mdcr/articles.jsonl" \
    --output_questions_file "${STORAGE_DATA}/qa/mdcr/questions.json" \
    --output_gold_contexts_file "${STORAGE_DATA}/qa/mdcr/questions.gold_contexts.json"
