#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED/DocRED_baseline_metadata
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/ed

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

for split in train dev test
do
    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/linked-docred/${split}.json \
        --entity_dict ${STORAGE_DATA}/../kb/dbpedia/dbpedia20200201.entity_dict.json
done

# Results:
#   - STORAGE_DATA/linked-docred/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/linked-docred/meta
cp ${DOCRED}/ner2id.json ${STORAGE_DATA}/linked-docred/meta/
