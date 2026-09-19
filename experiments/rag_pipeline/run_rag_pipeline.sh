#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/rag_pipeline/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/rag_pipeline/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
CONFIG_NAME=qwen3_embedding_06b___gpt5_4_nano

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/corpus/passages.jsonl
INPUT_QUESTIONS=${STORAGE_DATA}/examples/qa/questions.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_QUESTIONS=${STORAGE_DATA}/examples/qa/questions_with_answers.json
GOLD_CONTEXTS=${STORAGE_DATA}/examples/qa/questions.gold_contexts.json

# (optional) External Index
EXTERNAL_INDEX_DIR=

######
# Command-line arguments
######

# Initialize the requested action
ACTIONTYPE=
BATCH_MODE=

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--actiontype" ]; then
        if [ "$#" -lt 2 ]; then
            echo "Error: --actiontype requires a value"
            exit 1
        fi
        ACTIONTYPE=$2
        shift 2
    elif [ "$1" = "--batch_mode" ]; then
        if [ "$#" -lt 2 ]; then
            echo "Error: --batch_mode requires a value"
            exit 1
        fi
        BATCH_MODE=$2
        shift 2
    else
        echo "Error: Unknown argument: $1"
        exit 1
    fi
done

######
# Experiment execution
######

# Prepare the optional external index argument
EXTERNAL_INDEX_ARGS=()
if [ -n "${EXTERNAL_INDEX_DIR}" ]; then
    EXTERNAL_INDEX_ARGS+=(--external_index_dir "${EXTERNAL_INDEX_DIR}")
fi

# Prepare the optional Batch API arguments
BATCH_MODE_ARGS=()
if [ -n "${BATCH_MODE}" ]; then
    BATCH_MODE_ARGS=(--batch_mode "${BATCH_MODE}")
fi

if [ "${ACTIONTYPE}" = "indexing" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_passages ${INPUT_PASSAGES} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype indexing
fi

if [ "${ACTIONTYPE}" = "inference" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_questions ${INPUT_QUESTIONS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype inference \
        --do_evaluation \
        --gold_answers ${GOLD_QUESTIONS} \
        --gold_contexts ${GOLD_CONTEXTS} \
        "${EXTERNAL_INDEX_ARGS[@]}" \
        "${BATCH_MODE_ARGS[@]}"
fi
