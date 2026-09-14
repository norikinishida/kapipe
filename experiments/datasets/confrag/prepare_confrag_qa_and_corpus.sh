#!/usr/bin/env bash

set -euo pipefail


CONFRAG=/home/nishida/storage/dataset/ConfRAG
CONFRAG_REVISION=529e760a78a58791d0387fe469e0732377a6d94d
CONFRAG_FILENAME=ConfRAGsuggested.jsonl
CONFRAG_SHA256=bfe5fdc8e1e56300e36e7477626e118ba489469d971d9fc3f815ffd3e243bdcf
CONFRAG_URL="https://huggingface.co/datasets/OracleY/ConfRAG/resolve/${CONFRAG_REVISION}/${CONFRAG_FILENAME}"

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download the recommended official annotations from a fixed revision
# Results:
#   - CONFRAG/CONFRAG_FILENAME
mkdir -p "${CONFRAG}"
wget -c \
    "${CONFRAG_URL}" \
    -O "${CONFRAG}/${CONFRAG_FILENAME}"

# Validate the integrity of the downloaded file using its SHA-256 checksum
echo "${CONFRAG_SHA256}  ${CONFRAG}/${CONFRAG_FILENAME}" \
    | sha256sum --check --status

# Results:
#   - STORAGE_DATA/confrag/corpus/articles.jsonl
#   - STORAGE_DATA/confrag/qa/train.json
#   - STORAGE_DATA/confrag/qa/train.gold_contexts.json
python prepare_confrag_qa_and_corpus.py \
    --input_file "${CONFRAG}/${CONFRAG_FILENAME}" \
    --output_articles_file "${STORAGE_DATA}/confrag/corpus/articles.jsonl" \
    --output_dir "${STORAGE_DATA}/confrag/qa"
