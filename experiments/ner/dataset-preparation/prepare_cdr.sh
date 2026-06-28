#!/usr/bin/env bash

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph/data/CDR/processed

STORAGE=/home/nishida/storage/projects/kapipe/experiments/ner
STORAGE_DATA=${STORAGE}/data

# Results:
#   - STORAGE_DATA/cdr/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr.py \
        --input_file ${EOG}/${split}_filter.data \
        --output_file ${STORAGE_DATA}/cdr/${split}.json
done

# Results:
#   - STORAGE_DATA/cdr/demonstration_documents.json
N_DEMONSTRATIONS=3
python generate_demonstrations.py \
    --input_file ${STORAGE_DATA}/cdr/train.json \
    --n_demonstrations ${N_DEMONSTRATIONS} \
    --output_file ${STORAGE_DATA}/cdr/demonstration_documents.json

