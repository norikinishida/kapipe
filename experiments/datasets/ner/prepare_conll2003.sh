#!/usr/bin/env bash

CONLL03=/home/nishida/storage/dataset/CoNLL2003/NER/ner

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets

# Results:
#   - STORAGE_DATA/ner/conll2003/{train,dev,test}.json
#   - STORAGE_DATA/ner/conll2003/meta/entity_type_to_id.json
for split in train testa testb
do
    # Map official source splits to the output split names
    case ${split} in
        train) output_split=train ;;
        testa) output_split=dev ;;
        testb) output_split=test ;;
    esac
    python prepare_conll2003.py \
        --input_file ${CONLL03}/eng.${split} \
        --output_file ${STORAGE_DATA}/ner/conll2003/${output_split}.json \
        --split ${output_split}
done
