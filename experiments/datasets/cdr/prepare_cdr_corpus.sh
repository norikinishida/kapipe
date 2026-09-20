#!/usr/bin/env sh

# Note that the CDR directory is already generated using the script `experiments/datasets/cdr/prepare_cdr_ner.sh` before running this script.
CDR=/home/nishida/storage/projects/kapipe/experiments/datasets/cdr/docre

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/cdr/corpus/cdr_abstracts.jsonl
python prepare_cdr_corpus.py \
    --input_dir ${CDR} \
    --output_file ${STORAGE_DATA}/cdr/corpus/cdr_abstracts.jsonl

