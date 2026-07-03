#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/docre/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/docre/results

# STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/docre
# STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/docre/results

######
# Experiment configuration
######

# Method
# METHOD=atlop
METHOD=llm_docre

if [ "${METHOD}" == "atlop" ]; then
    CONFIG_PATH=./config/atlop.conf
    CONFIG_NAME=atlop_cdr
    # CONFIG_NAME=atlop_linked_docred
elif [ "${METHOD}" == "llm_docre" ]; then
    CONFIG_PATH=./config/llm_docre.conf
    # CONFIG_NAME=llm_docre_cdr
    # CONFIG_NAME=llm_docre_linked_docred
    CONFIG_NAME=llm_docre_user_defined
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents.ner.ed_ret.ed_rank.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_docre.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
