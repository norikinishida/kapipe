#!/usr/bin/env bash

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/pubmed/corpus/pubmed_abstracts.jsonl
python prepare_pubmed_corpus.py \
    --output_file ${STORAGE_DATA}/pubmed/corpus/pubmed_abstracts.jsonl

