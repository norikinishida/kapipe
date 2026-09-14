#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/docred/docre/{train,dev,test}.json
python prepare_docred_docre.py \
    --input_file ${DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/docred/docre/train.json \
    --split train

for split in dev test
do
    python prepare_docred_docre.py \
        --input_file ${DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/docred/docre/${split}.json \
        --split ${split}
done


# Results:
#   - STORAGE_DATA/docred/docre/meta/rel2id.json
#   - STORAGE_DATA/docred/docre/meta/rel_info.json
#   - STORAGE_DATA/docred/docre/meta/ner2id.json
#   - STORAGE_DATA/docred/docre/original/train_annotated.json
#   - STORAGE_DATA/docred/docre/original/train_distant.json
#   - STORAGE_DATA/docred/docre/original/dev.json
mkdir -p ${STORAGE_DATA}/docred/docre/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/docred/docre/meta/
cp ${DOCRED}/rel_info.json ${STORAGE_DATA}/docred/docre/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/docred/docre/meta/
mkdir -p ${STORAGE_DATA}/docred/docre/original
cp ${DOCRED}/train_annotated.json ${STORAGE_DATA}/docred/docre/original/
cp ${DOCRED}/train_distant.json ${STORAGE_DATA}/docred/docre/original/
cp ${DOCRED}/dev.json ${STORAGE_DATA}/docred/docre/original/
