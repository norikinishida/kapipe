#!/usr/bin/env bash

BIOASQ_TRAINDEV=/home/nishida/storage/dataset/BioASQ/BioASQ12/BioASQ-training12b/training12b_new.json

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets

SIZE=256


# Results
#   - STORAGE_DATA/qa/bioasq/train_dev.json
#   - STORAGE_DATA/qa/bioasq/train_dev.contexts.json
python prepare_bioasq.py \
    --input_file ${BIOASQ_TRAINDEV} \
    --output_file ${STORAGE_DATA}/qa/bioasq/train_dev.json


# Results
#   - STORAGE_DATA/qa/bioasq/train_dev_list_only_pubmed_limited_${SIZE}.json
#   - STORAGE_DATA/qa/bioasq/train_dev_list_only_pubmed_limited_${SIZE}.contexts.json
python prepare_bioasq.py \
    --input_file ${BIOASQ_TRAINDEV} \
    --output_file ${STORAGE_DATA}/qa/bioasq/train_dev_list_only_pubmed_limited_${SIZE}.json \
    --target_answer_types list \
    --pubmed_abstracts ${STORAGE_DATA}/articles/pubmed/pubmed_abstracts.jsonl \
    --size ${SIZE}


# Results
#   - STORAGE_DATA/qa/bioasq/passages.jsonl
python extract_passages.py \
    --input_file ${STORAGE_DATA}/qa/bioasq/train_dev.contexts.json \
    --output_file ${STORAGE_DATA}/qa/bioasq/passages.jsonl
