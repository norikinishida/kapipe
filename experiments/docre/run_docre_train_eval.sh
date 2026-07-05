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
METHOD=atlop
# METHOD=llm_docre

if [ "${METHOD}" == "atlop" ]; then
    CONFIG_PATH=./config/atlop.conf
    CONFIG_NAME=atlop_model_scibertcased_cdr_overlap
elif [ "${METHOD}" == "llm_docre" ]; then
    CONFIG_PATH=./config/llm_docre.conf
    CONFIG_NAME=gpt4omini_cdr
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

## Input Data
# (In practice, use separate files for training, validation, and test data.
# This example checks whether the model can achieve near 100% accuracy (i.e., overfit) on the training data.)
DATASET_NAME=cdr
TRAIN_DOCUMENTS=${STORAGE_DATA}/examples/documents_with_triples.json
DEV_DOCUMENTS=${STORAGE_DATA}/examples/documents_with_triples.json
TEST_DOCUMENTS=${STORAGE_DATA}/examples/documents_with_triples.json
#
ENTITY_DICT=${STORAGE_DATA}/examples/entity_dict.json
#
N_DEMONSTRATIONS=3

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

if [ "${METHOD}" == "atlop" ]; then
    python run_docre_train_eval.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --dataset_name ${DATASET_NAME} \
        --train_documents ${TRAIN_DOCUMENTS} \
        --dev_documents ${DEV_DOCUMENTS} \
        --test_documents ${TEST_DOCUMENTS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype train_and_evaluate
fi

if [ "${METHOD}" == "llm_docre" ]; then
    python run_docre_train_eval.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --dataset_name ${DATASET_NAME} \
        --train_documents ${TRAIN_DOCUMENTS} \
        --dev_documents ${DEV_DOCUMENTS} \
        --test_documents ${TEST_DOCUMENTS} \
        --n_demonstrations ${N_DEMONSTRATIONS} \
        --entity_dict ${ENTITY_DICT} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype evaluate
fi

