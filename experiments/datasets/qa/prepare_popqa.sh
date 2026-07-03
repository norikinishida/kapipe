#!/usr/bin/env bash

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/qa

SIZE=256


# Results
#   - STORAGE_DATA/popqa/test.json
#   - STORAGE_DATA/popqa/test_${SIZE}.json
python prepare_popqa.py \
    --output_dir ${STORAGE_DATA}/popqa \
    --size ${SIZE}

