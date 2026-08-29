#!/usr/bin/env bash

set -euo pipefail


STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/articles/longbench_v2/articles.jsonl
#   - STORAGE_DATA/qa/longbench_v2/train.json
#   - STORAGE_DATA/qa/longbench_v2/train.gold_contexts.json
python prepare_longbench_v2.py \
    --output_articles_file "${STORAGE_DATA}/articles/longbench_v2/articles.jsonl" \
    --output_dir "${STORAGE_DATA}/qa/longbench_v2"
