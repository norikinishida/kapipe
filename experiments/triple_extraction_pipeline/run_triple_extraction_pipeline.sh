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
CONFIG_NAME=llm_chemical_disease
# CONFIG_NAME=slm_chemical_disease

# Input Data
INPUT_DOCUMENTS=${STORAGE_DATA}/examples/documents.json
GOLD_DOCUMENTS=${STORAGE_DATA}/examples/documents_with_supervision.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

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
    --gold_documents ${GOLD_DOCUMENTS}

######
# Experiment on user-defined schema
#####

# CONFIG_NAME=llm_user_defined
# INPUT_DOCUMENTS=${STORAGE_DATA}/examples/documents2.json

# python run_triple_extraction_pipeline.py \
#     --method ${METHOD} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --input_documents ${INPUT_DOCUMENTS} \
#     --results_dir ${RESULTS_DIR} \
#     --prefix ${MYPREFIX}

