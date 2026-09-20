#!/usr/bin/env bash

DOCRED=/home/nishida/storage/dataset/DocRED
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/linked_docred/docre/{train,dev,test}.json
python prepare_linked_docred_docre.py \
    --input_file ${LINKED_DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/linked_docred/docre/train.json \
    --split train
for split in dev test
do
    python prepare_linked_docred_docre.py \
        --input_file ${LINKED_DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/linked_docred/docre/${split}.json \
        --split ${split}
done

# Results:
#   - STORAGE_DATA/linked_docred/docre/meta/rel2id.json
#   - STORAGE_DATA/linked_docred/docre/meta/rel_info.json
#   - STORAGE_DATA/linked_docred/docre/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/linked_docred/docre/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/linked_docred/docre/meta/
cp ${LINKED_DOCRED}/rel_info.json ${STORAGE_DATA}/linked_docred/docre/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/linked_docred/docre/meta/
