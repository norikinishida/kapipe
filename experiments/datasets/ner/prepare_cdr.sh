#!/usr/bin/env bash

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph/data/CDR/processed

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets

# Results:
#   - STORAGE_DATA/ner/cdr/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr.py \
        --input_file ${EOG}/${split}_filter.data \
        --output_file ${STORAGE_DATA}/ner/cdr/${split}.json
done
