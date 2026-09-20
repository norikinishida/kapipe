#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ed_retrieval/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ed_retrieval/results

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
TRAIN_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json
DEV_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json
TEST_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json
ENTITY_DICT=${STORAGE_DATA}/examples/kb/entity_dict.json

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
    --train_documents ${TRAIN_DOCUMENTS} \
    --dev_documents ${DEV_DOCUMENTS} \
    --test_documents ${TEST_DOCUMENTS} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype train_and_evaluate
