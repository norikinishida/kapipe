#!/usr/bin/env bash

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/articles


# Results:
#   - STORAGE_DATA/pubmed/pubmed_abstracts.jsonl
python prepare_pubmed_abstracts.py \
    --output_file ${STORAGE_DATA}/articles/pubmed/pubmed_abstracts.jsonl

