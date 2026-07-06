#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets
STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/mention_agnostic_docre_bionlp2024/results

######
# Experiment configuration
######

# Method
METHOD=ma_atlop
# METHOD=ma_qa

if [ "${METHOD}" == "ma_atlop" ]; then
    CONFIG_PATH=./config/ma_atlop.conf
    CONFIG_NAME=ma_atlop_model_scibertcased_cdr
elif [ "${METHOD}" == "ma_qa" ]; then
    CONFIG_PATH=./config/ma_qa.conf
    CONFIG_NAME=ma_qa_model_scibertcased_cdr
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

## Input Data
# (In practice, use separate files for training, validation, and test data.
# This example checks whether the model can achieve near 100% accuracy (i.e., overfit) on the training data.)
DATASET_NAME=cdr
TRAIN_DOCUMENTS=${STORAGE_DATA}/docre/${DATASET_NAME}/train.json
DEV_DOCUMENTS=${STORAGE_DATA}/docre/${DATASET_NAME}/dev.json
TEST_DOCUMENTS=${STORAGE_DATA}/docre/${DATASET_NAME}/test.json
ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

if [ "${METHOD}" == "ma_atlop" ]; then
    python run_ma_docre_train_eval.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --dataset_name ${DATASET_NAME} \
        --train_documents ${TRAIN_DOCUMENTS} \
        --dev_documents ${DEV_DOCUMENTS} \
        --test_documents ${TEST_DOCUMENTS} \
        --entity_dict ${ENTITY_DICT} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype train_and_evaluate
fi

if [ "${METHOD}" == "ma_qa" ]; then
    python run_ma_docre_train_eval.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --dataset_name ${DATASET_NAME} \
        --train_documents ${TRAIN_DOCUMENTS} \
        --dev_documents ${DEV_DOCUMENTS} \
        --test_documents ${TEST_DOCUMENTS} \
        --entity_dict ${ENTITY_DICT} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype train_and_evaluate
fi

