#!/usr/bin/env bash

WIKIPEDIA=/home/nishida/storage/dataset/Wikipedia

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - WIKIPEDIA/psgs_w100.tsv
mkdir -p ${WIKIPEDIA}/fb
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz -P ${WIKIPEDIA}/fb
gzip -d ${WIKIPEDIA}/fb/psgs_w100.tsv.gz


# Results:
#   - STORAGE_DATA/articles/wikipedia/psgs_w100.jsonl
python prepare_wikipedia_psgs_w100.py \
    --input_file ${WIKIPEDIA}/fb/psgs_w100.tsv \
    --output_file ${STORAGE_DATA}/articles/wikipedia_psgs_w100/psgs_w100.jsonl

