#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED
REDOCRED=/home/nishida/storage/dataset/Re-DocRED/Re-DocRED/data

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/redocred/docre/{train,dev,test}.json
for split in train dev test
do
    python prepare_redocred_docre.py \
        --input_file ${REDOCRED}/${split}_revised.json \
        --output_file ${STORAGE_DATA}/redocred/docre/${split}.json \
        --split ${split}
done

# Results:
#   - STORAGE_DATA/redocred/docre/meta/rel2id.json
#   - STORAGE_DATA/redocred/docre/meta/rel_info.json
#   - STORAGE_DATA/redocred/docre/meta/ner2id.json
#   - STORAGE_DATA/redocred/docre/original/train_revised.json
#   - STORAGE_DATA/redocred/docre/original/train_distant.json
#   - STORAGE_DATA/redocred/docre/original/dev_revised.json
#   - STORAGE_DATA/redocred/docre/original/test_revised.json
mkdir -p ${STORAGE_DATA}/redocred/docre/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/redocred/docre/meta/
cp ${DOCRED}/rel_info.json ${STORAGE_DATA}/redocred/docre/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/redocred/docre/meta/
mkdir -p ${STORAGE_DATA}/redocred/docre/original
cp ${REDOCRED}/train_revised.json ${STORAGE_DATA}/redocred/docre/original/
cp ${DOCRED}/train_distant.json ${STORAGE_DATA}/redocred/docre/original/
cp ${REDOCRED}/dev_revised.json ${STORAGE_DATA}/redocred/docre/original/
cp ${REDOCRED}/test_revised.json ${STORAGE_DATA}/redocred/docre/original/

