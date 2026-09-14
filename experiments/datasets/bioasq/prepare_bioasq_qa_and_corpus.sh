#!/usr/bin/env bash

BIOASQ_TRAINDEV=/home/nishida/storage/dataset/BioASQ/BioASQ12/BioASQ-training12b/training12b_new.json

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results
#   - STORAGE_DATA/bioasq/qa/train_dev.json
#   - STORAGE_DATA/bioasq/qa/train_dev.contexts.json
python prepare_bioasq_qa.py \
    --input_file ${BIOASQ_TRAINDEV} \
    --output_file ${STORAGE_DATA}/bioasq/qa/train_dev.json


# Results
#   - STORAGE_DATA/bioasq/qa/train_dev_list_only_pubmed_limited.json
#   - STORAGE_DATA/bioasq/qa/train_dev_list_only_pubmed_limited.contexts.json
python prepare_bioasq_qa.py \
    --input_file ${BIOASQ_TRAINDEV} \
    --output_file ${STORAGE_DATA}/bioasq/qa/train_dev_list_only_pubmed_limited.json \
    --target_answer_types list \
    --pubmed_abstracts ${STORAGE_DATA}/pubmed/corpus/pubmed_abstracts.jsonl


# Results
#   - STORAGE_DATA/bioasq/corpus/passages.jsonl
python prepare_bioasq_corpus.py \
    --input_file ${STORAGE_DATA}/bioasq/qa/train_dev.contexts.json \
    --output_file ${STORAGE_DATA}/bioasq/corpus/passages.jsonl
