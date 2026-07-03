#!/usr/bin/env bash

MEDMENTIONS=/home/nishida/storage/dataset/MedMentions/MedMentions

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/ner

# Results:
#   - STORAGE_DATA/medmentions/{train,dev,test}.json
#   - STORAGE_DATA/medmentions/meta/st21pv_semantic_types.json
python prepare_medmentions.py \
    --input_dir ${MEDMENTIONS} \
    --output_dir ${STORAGE_DATA}/medmentions
