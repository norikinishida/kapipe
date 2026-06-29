#!/usr/bin/env bash

######
# Storage paths
######

STORAGE=/home/nishida/projects/kapipe/experiments/ner
# STORAGE=/home/nishida/storage/projects/kapipe/experiments/ner

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

######
# Experiment configuration
######

# METHOD=biaffine_ner
METHOD=llm_ner

# Method
if [ "${METHOD}" == "biaffine_ner" ]; then
    CONFIG_PATH=./config/biaffine_ner.conf
    CONFIG_NAME=biaffine_ner_model_scibertuncased_cdr
elif [ "${METHOD}" == "llm_ner" ]; then
    CONFIG_PATH=./config/llm_ner.conf
    CONFIG_NAME=gpt4omini_cdr
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
# (In practice, use separate files for training, validation, and test data.
# This example checks whether the model can achieve near 100% accuracy (i.e., overfit) on the training data.)
DATASET_NAME=cdr
TRAIN_DOCS=${STORAGE_DATA}/examples/documents_with_supervision.json
DEV_DOCS=${STORAGE_DATA}/examples/documents_with_supervision.json
TEST_DOCS=${STORAGE_DATA}/examples/documents_with_supervision.json
# Optionally, you can also provide demonstration documents for LLM-based NER.
DEMO_DOCS=${STORAGE_DATA}/examples/demonstration_documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

if [ "${METHOD}" == "biaffine_ner" ]; then
    python run_ner_train_eval.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --dataset_name ${DATASET_NAME} \
        --train_documents ${TRAIN_DOCS} \
        --dev_documents ${DEV_DOCS} \
        --test_documents ${TEST_DOCS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype train_and_evaluate
fi

if [ "${METHOD}" == "llm_ner" ]; then
    python run_ner_train_eval.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --dataset_name ${DATASET_NAME} \
        --train_documents ${TRAIN_DOCS} \
        --dev_documents ${DEV_DOCS} \
        --test_documents ${TEST_DOCS} \
        --demonstration_documents ${DEMO_DOCS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype evaluate
fi
