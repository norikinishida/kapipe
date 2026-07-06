#!/usr/bin/env bash

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets

SIZE=256


# Results
#   - STORAGE_DATA/qa/popqa/test.json
#   - STORAGE_DATA/qa/popqa/test_${SIZE}.json
python prepare_popqa.py \
    --output_dir ${STORAGE_DATA}/qa/popqa \
    --size ${SIZE}

