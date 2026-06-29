#!/usr/bin/env bash

######
# Storage paths
######

STORAGE=/home/nishida/projects/kapipe/experiments/ed_retrieval
# STORAGE=/home/nishida/storage/projects/kapipe/experiments/ed_retrieval

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

######
# Experiment configuration
######

# Method
METHOD=blink_bi_encoder

CONFIG_PATH=./config/blink_bi_encoder.conf
CONFIG_NAME=blink_bi_encoder_model_scibertuncased_cdr

# Input Data
# (In practice, use separate files for training, validation, and test data.
# This example checks whether the model can achieve near 100% accuracy (i.e., overfit) on the training data.)
TRAIN_DOCS=${STORAGE_DATA}/examples/documents_with_supervision.json
DEV_DOCS=${STORAGE_DATA}/examples/documents_with_supervision.json
TEST_DOCS=${STORAGE_DATA}/examples/documents_with_supervision.json
ENTITY_DICT=${STORAGE_DATA}/examples/entity_dict.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_ed_retrieval_train_eval.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --train_documents ${TRAIN_DOCS} \
    --dev_documents ${DEV_DOCS} \
    --test_documents ${TEST_DOCS} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype train_and_evaluate
