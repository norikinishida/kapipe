#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED
REDOCRED=/home/nishida/storage/dataset/Re-DocRED/Re-DocRED/data

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/docre


# Results:
#   - STORAGE_DATA/redocred/{train,dev,test}.json
for split in train dev test
do
    python prepare_docred.py \
        --input_file ${REDOCRED}/${split}_revised.json \
        --output_file ${STORAGE_DATA}/redocred/${split}.json
done

# Results:
#   - STORAGE_DATA/redocred/meta/rel2id.json
#   - STORAGE_DATA/redocred/meta/rel_info.json
#   - STORAGE_DATA/redocred/meta/ner2id.json
#   - STORAGE_DATA/redocred/original/train_revised.json
#   - STORAGE_DATA/redocred/original/train_distant.json
#   - STORAGE_DATA/redocred/original/dev_revised.json
#   - STORAGE_DATA/redocred/original/test_revised.json
mkdir -p ${STORAGE_DATA}/redocred/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/redocred/meta/
cp ${DOCRED}/rel_info.json ${STORAGE_DATA}/redocred/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/redocred/meta/
mkdir -p ${STORAGE_DATA}/redocred/original
cp ${REDOCRED}/train_revised.json ${STORAGE_DATA}/redocred/original/
cp ${DOCRED}/train_distant.json ${STORAGE_DATA}/redocred/original/
cp ${REDOCRED}/dev_revised.json ${STORAGE_DATA}/redocred/original/
cp ${REDOCRED}/test_revised.json ${STORAGE_DATA}/redocred/original/


