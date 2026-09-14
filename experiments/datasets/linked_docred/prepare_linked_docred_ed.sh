#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED/DocRED_baseline_metadata
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets

# Results:
#   - STORAGE_DATA/linked_docred/ed/{train,dev,test}.json
python prepare_linked_docred_ed.py \
    --input_file ${LINKED_DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/linked_docred/ed/train.json \
    --split train
for split in dev test
do
    python prepare_linked_docred_ed.py \
        --input_file ${LINKED_DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/linked_docred/ed/${split}.json \
        --split ${split}
done

for split in train dev test
do
    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/linked_docred/ed/${split}.json \
        --entity_dict ${STORAGE_DATA}/dbpedia/kb/dbpedia20200201.entity_dict.json
done

# Results:
#   - STORAGE_DATA/linked_docred/ed/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/linked_docred/ed/meta
cp ${DOCRED}/ner2id.json ${STORAGE_DATA}/linked_docred/ed/meta/
