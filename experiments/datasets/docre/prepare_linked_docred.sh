#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/docre


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
#   - STORAGE_DATA/linked-docred/meta/rel2id.json
#   - STORAGE_DATA/linked-docred/meta/rel_info.json
#   - STORAGE_DATA/linked-docred/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/linked-docred/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/linked-docred/meta/
cp ${LINKED_DOCRED}/rel_info.json ${STORAGE_DATA}/linked-docred/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/linked-docred/meta/

