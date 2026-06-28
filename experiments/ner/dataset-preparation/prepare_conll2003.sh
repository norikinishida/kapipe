#!/usr/bin/env bash

CONLL03=/home/nishida/storage/dataset/CoNLL2003/NER/ner

STORAGE=/home/nishida/storage/projects/kapipe/experiments/ner
STORAGE_DATA=${STORAGE}/data

# Results:
#   - STORAGE_DATA/conll2003/{train,testa,testb}.json
#   - STORAGE_DATA/conll2003/meta/entity_type_to_id.json
for split in train testa testb
do
    python prepare_conll2003.py \
        --input_file ${CONLL03}/eng.${split} \
        --output_file ${STORAGE_DATA}/conll2003/${split}.json
done

# Results:
#   - STORAGE_DATA/conll2003/demonstration_documents.json
N_DEMONSTRATIONS=3
python generate_demonstrations.py \
    --input_file ${STORAGE_DATA}/conll2003/train.json \
    --n_demonstrations ${N_DEMONSTRATIONS} \
    --output_file ${STORAGE_DATA}/conll2003/demonstration_documents.json
