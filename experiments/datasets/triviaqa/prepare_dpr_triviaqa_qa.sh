#!/usr/bin/env bash

set -euo pipefail


TRIVIAQA=/home/nishida/storage/dataset/TriviaQA

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results
#   - TRIVIAQA/dpr/trivia-{train,dev,test}.qa.csv.gz
#   - TRIVIAQA/dpr/trivia-{train,dev,test}.qa.csv
mkdir -p "${TRIVIAQA}/dpr"
for split in train dev test
do
    wget -c \
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-${split}.qa.csv.gz" \
        -P "${TRIVIAQA}/dpr"
    if [[ ! -f "${TRIVIAQA}/dpr/trivia-${split}.qa.csv" ]]; then
        gzip -dk "${TRIVIAQA}/dpr/trivia-${split}.qa.csv.gz"
    fi
done


# Results
#   - STORAGE_DATA/triviaqa/qa/{train,dev,test}.json
for split in train dev test
do
    python prepare_dpr_triviaqa_qa.py \
        --input_file "${TRIVIAQA}/dpr/trivia-${split}.qa.csv" \
        --output_file "${STORAGE_DATA}/triviaqa/qa/${split}.json"
done


