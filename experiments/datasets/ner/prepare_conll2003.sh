#!/usr/bin/env bash

CONLL03=/home/nishida/storage/dataset/CoNLL2003/NER/ner

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets

# Results:
#   - STORAGE_DATA/ner/conll2003/{train,testa,testb}.json
#   - STORAGE_DATA/ner/conll2003/meta/entity_type_to_id.json
for split in train testa testb
do
    python prepare_conll2003.py \
        --input_file ${CONLL03}/eng.${split} \
        --output_file ${STORAGE_DATA}/ner/conll2003/${split}.json
done
