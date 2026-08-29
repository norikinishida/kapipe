#!/usr/bin/env bash

set -euo pipefail


STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download, decrypt, validate, and convert the official fixed benchmark
# Results:
#   - STORAGE_DATA/articles/browsecomp_plus/articles.jsonl
#   - STORAGE_DATA/qa/browsecomp_plus/test.json
#   - STORAGE_DATA/qa/browsecomp_plus/test.gold_contexts.json
#   - STORAGE_DATA/qa/browsecomp_plus/test.evidence_contexts.json
python prepare_browsecomp_plus.py \
    --output_articles_file "${STORAGE_DATA}/articles/browsecomp_plus/articles.jsonl" \
    --output_dir "${STORAGE_DATA}/qa/browsecomp_plus"
