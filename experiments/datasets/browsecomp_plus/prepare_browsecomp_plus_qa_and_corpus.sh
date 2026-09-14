#!/usr/bin/env bash

set -euo pipefail


STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download, decrypt, validate, and convert the official fixed benchmark
# Results:
#   - STORAGE_DATA/browsecomp_plus/corpus/articles.jsonl
#   - STORAGE_DATA/browsecomp_plus/qa/test.json
#   - STORAGE_DATA/browsecomp_plus/qa/test.gold_contexts.json
#   - STORAGE_DATA/browsecomp_plus/qa/test.evidence_contexts.json
python prepare_browsecomp_plus_qa_and_corpus.py \
    --output_articles_file "${STORAGE_DATA}/browsecomp_plus/corpus/articles.jsonl" \
    --output_dir "${STORAGE_DATA}/browsecomp_plus/qa"
