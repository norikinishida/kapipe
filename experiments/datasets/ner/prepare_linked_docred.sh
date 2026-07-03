#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED/DocRED_baseline_metadata
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/ner


# Results:
#   - STORAGE_DATA/linked_docred/{train,dev,test}.json
python prepare_linked_docred.py \
    --input_file ${LINKED_DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/linked_docred/train.json
for split in dev test
do
    python prepare_linked_docred.py \
        --input_file ${LINKED_DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/linked_docred/${split}.json
done

# Results:
#   - STORAGE_DATA/linked_docred/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/linked_docred/meta
cp ${DOCRED}/ner2id.json ${STORAGE_DATA}/linked_docred/meta/
