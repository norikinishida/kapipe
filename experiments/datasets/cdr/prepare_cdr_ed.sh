#!/usr/bin/env bash

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph/data/CDR/processed

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/cdr/ed/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr_ed.py \
        --input_file ${EOG}/${split}_filter.data \
        --output_file ${STORAGE_DATA}/cdr/ed/${split}.json \
        --split ${split}

    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/cdr/ed/${split}.json \
        --entity_dict ${STORAGE_DATA}/mesh/kb/mesh2015.entity_dict.json

done
