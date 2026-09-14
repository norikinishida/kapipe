#!/usr/bin/env bash

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph/data/CDR/processed

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/cdr/docre/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr_docre.py \
        --input_file ${EOG}/${split}_filter.data \
        --output_file ${STORAGE_DATA}/cdr/docre/${split}.json \
        --split ${split}
done
