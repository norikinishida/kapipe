#!/usr/bin/env bash

set -euo pipefail


NQ=/home/nishida/storage/dataset/NQ
TRIVIAQA=/home/nishida/storage/dataset/TriviaQA

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results
#   - NQ/fb/nq-{train,dev,test}.qa.csv
mkdir -p "${NQ}/dpr"
# for split in train dev test
for split in dev test
do
    wget -c \
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-${split}.qa.csv" \
        -P "${NQ}/dpr"
done


# Results
#   - STORAGE_DATA/qa/nq/{train,dev,test}.json
# for split in train dev test
for split in dev test
do
    python prepare_dpr_nq_triviaqa.py \
        --input_file "${NQ}/dpr/nq-${split}.qa.csv" \
        --output_file "${STORAGE_DATA}/qa/nq/${split}.json"
done


# Results
#   - TRIVIAQA/dpr/trivia-{train,dev,test}.qa.csv.gz
#   - TRIVIAQA/dpr/trivia-{train,dev,test}.qa.csv
mkdir -p "${TRIVIAQA}/dpr"
# for split in train dev test
for split in dev test
do
    wget -c \
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-${split}.qa.csv.gz" \
        -P "${TRIVIAQA}/dpr"
    if [[ ! -f "${TRIVIAQA}/dpr/trivia-${split}.qa.csv" ]]; then
        gzip -dk "${TRIVIAQA}/dpr/trivia-${split}.qa.csv.gz"
    fi
done


# Results
#   - STORAGE_DATA/qa/triviaqa/{train,dev,test}.json
# for split in train dev test
for split in dev test
do
    python prepare_dpr_nq_triviaqa.py \
        --input_file "${TRIVIAQA}/dpr/trivia-${split}.qa.csv" \
        --output_file "${STORAGE_DATA}/qa/triviaqa/${split}.json"
done


