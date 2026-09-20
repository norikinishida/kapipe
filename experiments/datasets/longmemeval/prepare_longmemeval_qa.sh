#!/usr/bin/env bash

set -euo pipefail


LONGMEMEVAL=/home/nishida/storage/dataset/LongMemEval

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download the two official cleaned history-length settings
# Results:
#   - LONGMEMEVAL/longmemeval_s_cleaned.json
#   - LONGMEMEVAL/longmemeval_m_cleaned.json
mkdir -p "${LONGMEMEVAL}"
wget -c \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
    -P "${LONGMEMEVAL}"
wget -c \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json \
    -P "${LONGMEMEVAL}"

# Results:
#   - STORAGE_DATA/longmemeval/qa/{small,medium}.json
#   - STORAGE_DATA/longmemeval/qa/{small,medium}.sessions.json
python prepare_longmemeval_qa.py \
    --input_small_file "${LONGMEMEVAL}/longmemeval_s_cleaned.json" \
    --input_medium_file "${LONGMEMEVAL}/longmemeval_m_cleaned.json" \
    --output_dir "${STORAGE_DATA}/longmemeval/qa"
