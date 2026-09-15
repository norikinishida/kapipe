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

# Input Artifacts
INPUT_PROPOSITIONS=
INPUT_TRIPLES=
INPUT_REFINED_TRIPLES=

# Input Index
INPUT_INDEX_DIR=

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

# Validate the action type
if [ -z "${ACTIONTYPE}" ]; then
    echo "Usage: bash run_prostruct_rag_pipeline.sh --actiontype {proposition_extraction|proposition_relation_extraction|proposition_relation_refinement|passage_graph_construction|passage_retrieval_indexing|inference|all} [--batch_mode {submit|fetch}]"
    exit 1
fi

# Validate the optional Batch API mode
if [ -n "${BATCH_MODE}" ] && [ "${BATCH_MODE}" != "submit" ] && [ "${BATCH_MODE}" != "fetch" ]; then
    echo "Error: --batch_mode must be submit or fetch"
    exit 1
fi

# Validate that "all" action type is not used with batch mode
if [ "${ACTIONTYPE}" = "all" ] && [ -n "${BATCH_MODE}" ]; then
    echo "Error: Run each stage separately with submit and fetch"
    exit 1
fi

######
# Experiment execution
######

# Prepare the optional input artifact arguments
INPUT_ARTIFACT_ARGS=()
if [ -n "${INPUT_PROPOSITIONS}" ]; then
    INPUT_ARTIFACT_ARGS+=(--input_propositions "${INPUT_PROPOSITIONS}")
fi
if [ -n "${INPUT_TRIPLES}" ]; then
    INPUT_ARTIFACT_ARGS+=(--input_triples "${INPUT_TRIPLES}")
fi
if [ -n "${INPUT_REFINED_TRIPLES}" ]; then
    INPUT_ARTIFACT_ARGS+=(--input_refined_triples "${INPUT_REFINED_TRIPLES}")
fi

# Prepare the optional input index argument
INPUT_INDEX_ARGS=()
if [ -n "${INPUT_INDEX_DIR}" ]; then
    INPUT_INDEX_ARGS+=(--input_index_dir "${INPUT_INDEX_DIR}")
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
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype proposition_relation_extraction \
        "${INPUT_ARTIFACT_ARGS[@]}" \
        "${BATCH_MODE_ARGS[@]}"
fi

if [ "${ACTIONTYPE}" = "proposition_relation_refinement" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype proposition_relation_refinement \
        "${INPUT_ARTIFACT_ARGS[@]}" \
        "${BATCH_MODE_ARGS[@]}"
fi

if [ "${ACTIONTYPE}" = "passage_graph_construction" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype passage_graph_construction \
        "${INPUT_ARTIFACT_ARGS[@]}"
fi

if [ "${ACTIONTYPE}" = "passage_retrieval_indexing" ] || [ "${ACTIONTYPE}" = "all" ]; then
    python run_prostruct_rag_pipeline.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype passage_retrieval_indexing \
        "${INPUT_ARTIFACT_ARGS[@]}"
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
        "${INPUT_INDEX_ARGS[@]}" \
        "${BATCH_MODE_ARGS[@]}"
fi
