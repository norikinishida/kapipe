#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/prostruct_rag_pipeline_emnlp2026/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/prostruct_rag_pipeline_emnlp2026/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
CONFIG_NAME=gpt4o_mini___contriever_top20___gpt4o_mini_temporal___gpt4o_temporal___default___contriever_top10___hop0_temporal___temporal___gpt4o_with_context

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/corpus/articles.jsonl
INPUT_QUESTIONS=${STORAGE_DATA}/examples/qa/questions.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_QUESTIONS=${STORAGE_DATA}/examples/qa/questions.json

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

if [ "${ACTIONTYPE}" = "proposition_extraction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_passages ${INPUT_PASSAGES} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype proposition_extraction \
        "${BATCH_MODE_ARGS[@]}"
fi

if [ "${ACTIONTYPE}" = "proposition_relation_extraction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_passages ${INPUT_PASSAGES} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype proposition_relation_extraction \
        "${BATCH_MODE_ARGS[@]}"
fi

if [ "${ACTIONTYPE}" = "proposition_relation_refinement" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_passages ${INPUT_PASSAGES} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype proposition_relation_refinement \
        "${BATCH_MODE_ARGS[@]}"
fi

if [ "${ACTIONTYPE}" = "passage_graph_construction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_passages ${INPUT_PASSAGES} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype passage_graph_construction
fi

if [ "${ACTIONTYPE}" = "passage_retrieval_indexing" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_passages ${INPUT_PASSAGES} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype passage_retrieval_indexing
fi

if [ "${ACTIONTYPE}" = "inference" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_questions ${INPUT_QUESTIONS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype inference \
        --do_evaluation \
        --gold ${GOLD_QUESTIONS} \
        "${EXTERNAL_INDEX_ARGS[@]}" \
        "${BATCH_MODE_ARGS[@]}"
fi
