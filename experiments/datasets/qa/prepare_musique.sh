#!/usr/bin/env bash

set -euo pipefail


STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/qa/musique/{train,dev}.json
#   - STORAGE_DATA/qa/musique/{train,dev}.gold_contexts.json
#   - STORAGE_DATA/qa/musique/{train,dev}.gold_contexts_with_distractors.json
python prepare_musique.py \
    --output_dir "${STORAGE_DATA}/qa/musique"
