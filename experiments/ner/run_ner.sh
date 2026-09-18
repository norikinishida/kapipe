#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ner/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ner/results

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
INPUT_DOCUMENTS=${STORAGE_DATA}/examples/ner/documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_DOCUMENTS=${STORAGE_DATA}/examples/ner/documents_with_typed_mentions.json

######
# Experiment execution
######

python run_ner.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents ${INPUT_DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --do_evaluation \
    --gold ${GOLD_DOCUMENTS} \
    "$@"
