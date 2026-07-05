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
INPUT_DOCUMENTS=${STORAGE_DATA}/examples/documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_DOCUMENTS=${STORAGE_DATA}/examples/documents_with_triples.json

######
# Experiment execution
######

python run_triple_extraction_pipeline.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents ${INPUT_DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --do_evaluation \
    --gold ${GOLD_DOCUMENTS}
