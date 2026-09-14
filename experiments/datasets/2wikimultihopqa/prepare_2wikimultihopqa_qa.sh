#!/usr/bin/env bash

set -euo pipefail


STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/2wikimultihopqa/qa/{train,dev}.json
#   - STORAGE_DATA/2wikimultihopqa/qa/{train,dev}.gold_contexts.json
#   - STORAGE_DATA/2wikimultihopqa/qa/{train,dev}.gold_contexts_at_sentence_level.json
#   - STORAGE_DATA/2wikimultihopqa/qa/{train,dev}.gold_contexts_with_distractors.json
python prepare_2wikimultihopqa_qa.py \
    --output_dir "${STORAGE_DATA}/2wikimultihopqa/qa"
