#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
GPU=0
METHOD=contriever
METRIC=inner-product
TOP_K=3

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/reports.chunked_w100.jsonl
INPUT_QUESTIONS=${STORAGE_DATA}/examples/questions.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
INDEX_NAME=example

python run_passage_retrieval.py \
    --gpu ${GPU} \
    --method ${METHOD} \
    --metric ${METRIC} \
    --input_file ${INPUT_PASSAGES} \
    --results_dir ${RESULTS_DIR} \
    --index_name ${INDEX_NAME} \
    --actiontype indexing

python run_passage_retrieval.py \
    --gpu ${GPU} \
    --method ${METHOD} \
    --metric ${METRIC} \
    --top_k ${TOP_K} \
    --input_file ${INPUT_QUESTIONS} \
    --results_dir ${RESULTS_DIR} \
    --index_name ${INDEX_NAME} \
    --actiontype search
