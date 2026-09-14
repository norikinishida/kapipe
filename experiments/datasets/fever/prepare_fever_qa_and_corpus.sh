#!/usr/bin/env bash

set -euo pipefail


FEVER_URL=https://fever.ai/download/fever
TRAIN_FILENAME=train.jsonl
DEV_FILENAME=shared_task_dev.jsonl
TRAIN_MD5=2216e2f367e223bf3e593a119222d99d
DEV_MD5=4eecc1018b3d2bd46089ca36d886f439

FEVER=/home/nishida/storage/dataset/FEVER

WIKIPEDIA_FILENAME=wiki-pages.zip
WIKIPEDIA_MD5=ed8bfd894a2c47045dca61f0c8dc4c07

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download the official labelled FEVER training and development splits
# Results:
#   - FEVER/TRAIN_FILENAME
#   - FEVER/DEV_FILENAME
mkdir -p "${FEVER}"
wget -c \
    "${FEVER_URL}/${TRAIN_FILENAME}" \
    -O "${FEVER}/${TRAIN_FILENAME}"
wget -c \
    "${FEVER_URL}/${DEV_FILENAME}" \
    -O "${FEVER}/${DEV_FILENAME}"

# Download the official pre-processed June 2017 English Wikipedia dump
# Results:
#   - FEVER/WIKIPEDIA_FILENAME
wget -c \
    "${FEVER_URL}/${WIKIPEDIA_FILENAME}" \
    -O "${FEVER}/${WIKIPEDIA_FILENAME}"

# Validate the integrity of the downloaded files using their MD5 checksums
echo "${TRAIN_MD5}  ${FEVER}/${TRAIN_FILENAME}" \
    | md5sum --check --status
echo "${DEV_MD5}  ${FEVER}/${DEV_FILENAME}" \
    | md5sum --check --status
echo "${WIKIPEDIA_MD5}  ${FEVER}/${WIKIPEDIA_FILENAME}" \
    | md5sum --check --status

# Extract the sharded official Wikipedia JSONL files once
# Results:
#   - FEVER/wiki-pages/*.jsonl
if [[ ! -d "${FEVER}/wiki-pages" ]]; then
    unzip -q "${FEVER}/${WIKIPEDIA_FILENAME}" -d "${FEVER}"
fi

# Convert the corpus, questions, labels, and alternative gold evidence sets
# Results:
#   - STORAGE_DATA/fever/corpus/articles.jsonl
#   - STORAGE_DATA/fever/qa/{train,dev}.json
#   - STORAGE_DATA/fever/qa/{train,dev}.gold_contexts_sets.json
python prepare_fever_qa_and_corpus.py \
    --input_train_file "${FEVER}/${TRAIN_FILENAME}" \
    --input_dev_file "${FEVER}/${DEV_FILENAME}" \
    --input_wikipedia_dir "${FEVER}/wiki-pages" \
    --output_articles_file "${STORAGE_DATA}/fever/corpus/articles.jsonl" \
    --output_dir "${STORAGE_DATA}/fever/qa"
