#!/usr/bin/env sh

HOIP=/home/nishida/projects/hoip-dataset/releases/v1

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/hoip/kb/hoip.entity_dict.json
python prepare_hoip_kb.py \
    --input_file ${HOIP}/hoip_ontology.json \
    --output_file ${STORAGE_DATA}/hoip/kb/hoip.entity_dict.json

