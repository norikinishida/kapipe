#!/usr/bin/env bash

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph/data/CDR/processed

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/ed


# Results:
#   - STORAGE_DATA/cdr/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr.py \
        --input_file ${EOG}/${split}_filter.data \
        --output_file ${STORAGE_DATA}/cdr/${split}.json

    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/cdr/${split}.json \
        --entity_dict ${STORAGE_DATA}/../kb/mesh/mesh2015.entity_dict.json

done
