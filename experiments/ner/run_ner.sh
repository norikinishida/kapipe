#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ner/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ner/results

# STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/ner
# STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/ner/results

######
# Experiment configuration
######

# Method
# METHOD=biaffine_ner
METHOD=llm_ner

if [ "${METHOD}" == "biaffine_ner" ]; then
    CONFIG_PATH=./config/biaffine_ner.conf
    CONFIG_NAME=biaffine_ner_cdr
    # CONFIG_NAME=biaffine_ner_linked_docred
elif [ "${METHOD}" == "llm_ner" ]; then
    CONFIG_PATH=./config/llm_ner.conf
    CONFIG_NAME=llm_ner_cdr
    # CONFIG_NAME=llm_ner_linked_docred
    # CONFIG_NAME=llm_ner_user_defined
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_ner.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
