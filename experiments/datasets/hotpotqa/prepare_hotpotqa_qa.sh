#!/usr/bin/env bash

set -euo pipefail


STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/hotpotqa/qa/{train,dev}.json
#   - STORAGE_DATA/hotpotqa/qa/{train,dev}.gold_contexts.json
#   - STORAGE_DATA/hotpotqa/qa/{train,dev}.gold_contexts_at_sentence_level.json
#   - STORAGE_DATA/hotpotqa/qa/{train,dev}.gold_contexts_with_distractors.json
python prepare_hotpotqa_qa.py \
    --output_dir "${STORAGE_DATA}/hotpotqa/qa"
