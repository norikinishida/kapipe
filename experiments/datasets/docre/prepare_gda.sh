#!/usr/bin/env bash

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph/data/GDA/processed

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/docre


# Results:
#   - STORAGE_DATA/gda/{train,dev,test}.json
for split in train dev test
do
    python prepare_gda.py \
        --input_file ${EOG}/${split}.data \
        --output_file ${STORAGE_DATA}/gda/${split}.json
done
