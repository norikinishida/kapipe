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
METHOD=default
CONFIG_PATH=./config/default.conf
CONFIG_NAME=qwen3_embedding_06b___gpt5_4_nano

# Input Data
INPUT_QUESTIONS=${STORAGE_DATA}/examples/questions.json
GOLD_QUESTIONS=${STORAGE_DATA}/examples/questions_with_answers.json
GOLD_CONTEXTS=${STORAGE_DATA}/examples/questions.gold_contexts.json

# Please run the `run_passage_retrieval_indexing.sh` script to generate the index before running this script.
INDEX_DIR=${STORAGE_RESULTS}/passage_retrieval/qwen3_embedding/qwen3_embedding_06b/example/indexes

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_tool_calling_agent.py \
    --method "${METHOD}" \
    --config_path "${CONFIG_PATH}" \
    --config_name "${CONFIG_NAME}" \
    --input_questions "${INPUT_QUESTIONS}" \
    --index_dir ${INDEX_DIR} \
    --results_dir "${RESULTS_DIR}" \
    --prefix "${MYPREFIX}" \
    --do_evaluation \
    --gold_answers "${GOLD_QUESTIONS}" \
    --gold_contexts "${GOLD_CONTEXTS}"
