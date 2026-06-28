#!/usr/bin/env bash

MEDMENTIONS=/home/nishida/storage/dataset/MedMentions/MedMentions

STORAGE=/home/nishida/storage/projects/kapipe/experiments/ner
STORAGE_DATA=${STORAGE}/data

# Results:
#   - STORAGE_DATA/medmentions/{train,dev,test}.json
#   - STORAGE_DATA/medmentions/meta/st21pv_semantic_types.json
python prepare_medmentions.py \
    --input_dir ${MEDMENTIONS} \
    --output_dir ${STORAGE_DATA}/medmentions

# Results:
#   - STORAGE_DATA/medmentions/demonstration_documents.json
N_DEMONSTRATIONS=3
python generate_demonstrations.py \
    --input_file ${STORAGE_DATA}/medmentions/train.json \
    --n_demonstrations ${N_DEMONSTRATIONS} \
    --output_file ${STORAGE_DATA}/medmentions/demonstration_documents.json
