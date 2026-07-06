#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED
REDOCRED=/home/nishida/storage/dataset/Re-DocRED/Re-DocRED/data

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/docre/redocred/{train,dev,test}.json
for split in train dev test
do
    python prepare_docred.py \
        --input_file ${REDOCRED}/${split}_revised.json \
        --output_file ${STORAGE_DATA}/docre/redocred/${split}.json
done

# Results:
#   - STORAGE_DATA/docre/redocred/meta/rel2id.json
#   - STORAGE_DATA/docre/redocred/meta/rel_info.json
#   - STORAGE_DATA/docre/redocred/meta/ner2id.json
#   - STORAGE_DATA/docre/redocred/original/train_revised.json
#   - STORAGE_DATA/docre/redocred/original/train_distant.json
#   - STORAGE_DATA/docre/redocred/original/dev_revised.json
#   - STORAGE_DATA/docre/redocred/original/test_revised.json
mkdir -p ${STORAGE_DATA}/docre/redocred/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/docre/redocred/meta/
cp ${DOCRED}/rel_info.json ${STORAGE_DATA}/docre/redocred/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/docre/redocred/meta/
mkdir -p ${STORAGE_DATA}/docre/redocred/original
cp ${REDOCRED}/train_revised.json ${STORAGE_DATA}/docre/redocred/original/
cp ${DOCRED}/train_distant.json ${STORAGE_DATA}/docre/redocred/original/
cp ${REDOCRED}/dev_revised.json ${STORAGE_DATA}/docre/redocred/original/
cp ${REDOCRED}/test_revised.json ${STORAGE_DATA}/docre/redocred/original/


