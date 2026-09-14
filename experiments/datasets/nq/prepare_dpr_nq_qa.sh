#!/usr/bin/env bash

set -euo pipefail


NQ=/home/nishida/storage/dataset/NQ

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results
#   - NQ/fb/nq-{train,dev,test}.qa.csv
mkdir -p "${NQ}/dpr"
for split in train dev test
do
    wget -c \
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-${split}.qa.csv" \
        -P "${NQ}/dpr"
done

# Results
#   - STORAGE_DATA/nq/qa/{train,dev,test}.json
for split in train dev test
do
    python prepare_dpr_nq_qa.py \
        --input_file "${NQ}/dpr/nq-${split}.qa.csv" \
        --output_file "${STORAGE_DATA}/nq/qa/${split}.json"
done
