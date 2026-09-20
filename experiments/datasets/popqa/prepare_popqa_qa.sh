#!/usr/bin/env bash

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results
#   - STORAGE_DATA/popqa/qa/test.json
python prepare_popqa_qa.py \
    --output_dir ${STORAGE_DATA}/popqa/qa
