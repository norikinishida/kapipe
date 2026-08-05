#!/usr/bin/env bash

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results
#   - STORAGE_DATA/qa/popqa/test.json
python prepare_popqa.py \
    --output_dir ${STORAGE_DATA}/qa/popqa

