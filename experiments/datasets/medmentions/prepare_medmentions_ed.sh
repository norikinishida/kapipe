#!/usr/bin/env bash

MEDMENTIONS=/home/nishida/storage/dataset/MedMentions/MedMentions

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/medmentions/ed/{train,dev,test}.json
#   - STORAGE_DATA/medmentions/ed/meta/st21pv_semantic_types.json
python prepare_medmentions_ed.py \
    --input_dir ${MEDMENTIONS} \
    --output_dir ${STORAGE_DATA}/medmentions/ed

for split in train dev test
do
    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/medmentions/ed/${split}.json \
        --entity_dict ${STORAGE_DATA}/umls/kb/umls2017aa.entity_dict.json
done

