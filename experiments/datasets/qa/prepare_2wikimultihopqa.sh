#!/usr/bin/env bash

set -euo pipefail


STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/qa/2wikimultihopqa/{train,dev}.json
#   - STORAGE_DATA/qa/2wikimultihopqa/{train,dev}.gold_contexts.json
#   - STORAGE_DATA/qa/2wikimultihopqa/{train,dev}.gold_contexts_at_sentence_level.json
#   - STORAGE_DATA/qa/2wikimultihopqa/{train,dev}.gold_contexts_with_distractors.json
python prepare_2wikimultihopqa.py \
    --output_dir "${STORAGE_DATA}/qa/2wikimultihopqa"
