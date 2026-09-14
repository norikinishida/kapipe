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
CONFIG_NAME=gpt4o_mini___gpt4o_mini_contriever_top20_temporal___gpt4o_temporal___default___contriever_top10___hop0_temporal___temporal___gpt4o_with_context

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/corpus/articles.jsonl
INPUT_QUESTIONS=${STORAGE_DATA}/examples/qa/questions.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_QUESTIONS=${STORAGE_DATA}/examples/qa/questions.json

######
# Command-line arguments
######

# Initialize the requested action
ACTIONTYPE=

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--actiontype" ]; then
        if [ "$#" -lt 2 ]; then
            echo "Error: --actiontype requires a value"
            exit 1
        fi
        ACTIONTYPE=$2
        shift 2
    else
        echo "Error: Unknown argument: $1"
        exit 1
    fi
done

# Validate the action type
if [ -z "${ACTIONTYPE}" ]; then
    echo "Usage: bash run_prostruct_rag_pipeline.sh --actiontype {proposition_extraction|proposition_relation_extraction|proposition_relation_refinement|passage_graph_construction|passage_retrieval_indexing|inference|all}"
    exit 1
fi

######
# Experiment execution
######

if [ "${ACTIONTYPE}" = "proposition_extraction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_passages ${INPUT_PASSAGES} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype proposition_extraction
fi

if [ "${ACTIONTYPE}" = "proposition_relation_extraction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype proposition_relation_extraction
fi

if [ "${ACTIONTYPE}" = "proposition_relation_refinement" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype proposition_relation_refinement
fi

if [ "${ACTIONTYPE}" = "passage_graph_construction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype passage_graph_construction
fi

if [ "${ACTIONTYPE}" = "passage_retrieval_indexing" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
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
        --gold ${GOLD_QUESTIONS}
fi
