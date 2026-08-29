#!/usr/bin/env bash

set -euo pipefail


LOONG=/home/nishida/storage/dataset/Loong
LOONG_REPOSITORY="${LOONG}/Loong"
LOONG_COMMIT=6d2115b8b48a3d19412ccb52a0c9c4ee37869af4
LOONG_DOCUMENTS_URL="http://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip"

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download the official repository only when it is not already available
# Results:
#   - LOONG_REPOSITORY/data/loong.jsonl
if [ ! -d "${LOONG_REPOSITORY}/.git" ]; then
    mkdir -p "${LOONG}"
    git clone \
        https://github.com/MozerWang/Loong.git \
        "${LOONG_REPOSITORY}"
    git -C "${LOONG_REPOSITORY}" \
        switch --detach "${LOONG_COMMIT}"
fi

# Require the tested dataset revision without modifying an existing clone
CURRENT_LOONG_COMMIT=$(
    git -C "${LOONG_REPOSITORY}" rev-parse HEAD
)
if [ "${CURRENT_LOONG_COMMIT}" != "${LOONG_COMMIT}" ]; then
    echo "Unexpected Loong revision: ${CURRENT_LOONG_COMMIT}" >&2
    echo "Expected revision: ${LOONG_COMMIT}" >&2
    exit 1
fi

# Download and extract the separately released source documents
# Results:
#   - LOONG_REPOSITORY/data/doc/{financial,legal,paper}
if [ ! -d "${LOONG_REPOSITORY}/data/doc" ]; then
    wget -c \
        "${LOONG_DOCUMENTS_URL}" \
        -O "${LOONG_REPOSITORY}/data/doc.zip"
    unzip \
        "${LOONG_REPOSITORY}/data/doc.zip" \
        -d "${LOONG_REPOSITORY}/data"
fi

# Results:
#   - STORAGE_DATA/articles/loong/{financial,legal,paper}_{en,zh}.articles.jsonl
#   - STORAGE_DATA/qa/loong/{financial,legal,paper}_{en,zh}.json
#   - STORAGE_DATA/qa/loong/{financial,legal,paper}_{en,zh}.gold_contexts.json
python prepare_loong.py \
    --input_questions_file "${LOONG_REPOSITORY}/data/loong.jsonl" \
    --input_documents_dir "${LOONG_REPOSITORY}/data/doc" \
    --output_articles_dir "${STORAGE_DATA}/articles/loong" \
    --output_qa_dir "${STORAGE_DATA}/qa/loong"
