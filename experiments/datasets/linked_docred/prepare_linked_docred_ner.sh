#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED/DocRED_baseline_metadata
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/linked_docred/ner/{train,dev,test}.json
python prepare_linked_docred_ner.py \
    --input_file ${LINKED_DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/linked_docred/ner/train.json \
    --split train
for split in dev test
do
    python prepare_linked_docred_ner.py \
        --input_file ${LINKED_DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/linked_docred/ner/${split}.json \
        --split ${split}
done

# Results:
#   - STORAGE_DATA/linked_docred/ner/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/linked_docred/ner/meta
cp ${DOCRED}/ner2id.json ${STORAGE_DATA}/linked_docred/ner/meta/
