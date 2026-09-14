#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/graphrag_pipeline_tacl2026/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/graphrag_pipeline_tacl2026/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
# CONFIG_NAME=slm___egc___na___temp___w100___contriever___gpt4o___cdr
CONFIG_NAME=llm___egc___hl___llm___w100___contriever___gpt4o___cdr

# Input Data
INPUT_DOCUMENTS=${STORAGE_DATA}/examples/docre/documents.json
ENTITY_DICT=${STORAGE_DATA}/examples/kb/entity_dict.json
INPUT_QUESTIONS=${STORAGE_DATA}/examples/qa/questions.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_QUESTIONS=${STORAGE_DATA}/examples/qa/questions_with_answers.json

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

# Validate that the action type is provided
if [ -z "${ACTIONTYPE}" ]; then
    echo "Usage: bash run_graphrag_pipeline.sh --actiontype {triple_extraction|entity_graph_construction|community_clustering|report_generation|chunking|passage_retrieval_indexing|inference|all}"
    exit 1
fi

######
# Experiment execution
######

if [ "${ACTIONTYPE}" = "triple_extraction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_graphrag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --input_documents ${INPUT_DOCUMENTS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype triple_extraction
fi

if [ "${ACTIONTYPE}" = "entity_graph_construction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_graphrag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --entity_dict ${ENTITY_DICT} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype entity_graph_construction
fi

if [ "${ACTIONTYPE}" = "community_clustering" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_graphrag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype community_clustering
fi

if [ "${ACTIONTYPE}" = "report_generation" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_graphrag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype report_generation
fi

if [ "${ACTIONTYPE}" = "chunking" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_graphrag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype chunking
fi

if [ "${ACTIONTYPE}" = "passage_retrieval_indexing" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_graphrag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype passage_retrieval_indexing
fi

if [ "${ACTIONTYPE}" = "inference" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_graphrag_pipeline.py \
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
