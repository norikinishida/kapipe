#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/triple_extraction_pipeline/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/triple_extraction_pipeline/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
# CONFIG_NAME=slm_cdr
CONFIG_NAME=llm_cdr
# CONFIG_NAME=llm_user_defined

# Input Data
INPUT_DOCUMENTS=${STORAGE_DATA}/examples/docre/documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_DOCUMENTS=${STORAGE_DATA}/examples/docre/documents_with_triples.json

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

# Prepare the optional Batch API arguments
BATCH_MODE_ARGS=()
if [ -n "${BATCH_MODE}" ]; then
    BATCH_MODE_ARGS=(--batch_mode "${BATCH_MODE}")
fi

python run_triple_extraction_pipeline.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents ${INPUT_DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype ${ACTIONTYPE} \
    --do_evaluation \
    --gold ${GOLD_DOCUMENTS} \
    "${BATCH_MODE_ARGS[@]}"
