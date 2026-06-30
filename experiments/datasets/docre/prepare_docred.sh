#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/docre


# Results:
#   - STORAGE_DATA/docred/{train,dev,test}.json
python prepare_docred.py \
    --input_file ${DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/docred/train.json

for split in dev test
do
    python prepare_docred.py \
        --input_file ${DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/docred/${split}.json
done


# Results:
#   - STORAGE_DATA/docred/meta/rel2id.json
#   - STORAGE_DATA/docred/meta/rel_info.json
#   - STORAGE_DATA/docred/meta/ner2id.json
#   - STORAGE_DATA/docred/original/train_annotated.json
#   - STORAGE_DATA/docred/original/train_distant.json
#   - STORAGE_DATA/docred/original/dev.json
mkdir -p ${STORAGE_DATA}/docred/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/docred/meta/
cp ${DOCRED}/rel_info.json ${STORAGE_DATA}/docred/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/docred/meta/
mkdir -p ${STORAGE_DATA}/docred/original
cp ${DOCRED}/train_annotated.json ${STORAGE_DATA}/docred/original/
cp ${DOCRED}/train_distant.json ${STORAGE_DATA}/docred/original/
cp ${DOCRED}/dev.json ${STORAGE_DATA}/docred/original/

