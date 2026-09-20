#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/agentic_search/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/agentic_search/results

######
# Experiment configuration
######

# Method
METHOD=qwen3_embedding
CONFIG_PATH=./config/default.conf
CONFIG_NAME=qwen3_embedding_06b

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/corpus/passages.jsonl

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_passage_retrieval_indexing.py \
    --method "${METHOD}" \
    --config_path "${CONFIG_PATH}" \
    --config_name "${CONFIG_NAME}" \
    --input_passages "${INPUT_PASSAGES}" \
    --results_dir "${RESULTS_DIR}" \
    --prefix "${MYPREFIX}"