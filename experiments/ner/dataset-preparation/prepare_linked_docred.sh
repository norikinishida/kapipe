#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED/DocRED_baseline_metadata
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

STORAGE=/home/nishida/storage/projects/kapipe/experiments/ner
STORAGE_DATA=${STORAGE}/data

# Results:
#   - STORAGE_DATA/linked-docred/{train,dev,test}.json
python prepare_linked_docred.py \
    --input_file ${LINKED_DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/linked-docred/train.json
for split in dev test
do
    python prepare_linked_docred.py \
        --input_file ${LINKED_DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/linked-docred/${split}.json
done

# Results:
#   - STORAGE_DATA/linked-docred/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/linked-docred/meta
cp ${DOCRED}/ner2id.json ${STORAGE_DATA}/linked-docred/meta/

# Results:
#   - STORAGE_DATA/cdr/demonstration_documents.json
N_DEMONSTRATIONS=3
python generate_demonstrations.py \
    --input_file ${STORAGE_DATA}/linked-docred/train.json \
    --n_demonstrations ${N_DEMONSTRATIONS} \
    --output_file ${STORAGE_DATA}/linked-docred/demonstration_documents.json
