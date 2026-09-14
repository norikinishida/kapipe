#!/usr/bin/env bash

set -euo pipefail


STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/longbench_v2/corpus/articles.jsonl
#   - STORAGE_DATA/longbench_v2/qa/train.json
#   - STORAGE_DATA/longbench_v2/qa/train.gold_contexts.json
python prepare_longbench_v2_qa_and_corpus.py \
    --output_articles_file "${STORAGE_DATA}/longbench_v2/corpus/articles.jsonl" \
    --output_dir "${STORAGE_DATA}/longbench_v2/qa"
