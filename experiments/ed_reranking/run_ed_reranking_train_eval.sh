#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ed_reranking/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ed_reranking/results

######
# Experiment configuration
######

# Method
METHOD=blink_cross_encoder
# METHOD=llm_ed

if [ "${METHOD}" == "blink_cross_encoder" ]; then
    CONFIG_PATH=./config/blink_cross_encoder.conf
    CONFIG_NAME=blink_cross_encoder_model_scibertuncased_cdr
elif [ "${METHOD}" == "llm_ed" ]; then
    CONFIG_PATH=./config/llm_ed.conf
    CONFIG_NAME=gpt4omini_cdr
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
# (In practice, use separate files for training, validation, and test data.
# This example checks whether the model can achieve near 100% accuracy (i.e., overfit) on the training data.)
TRAIN_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json
DEV_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json
TEST_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json
# For the sake of example, we use candidate entities and entity dictionary under `data/examples/`, but in actual experiments, please use the results (candidate entities and entity dictionary) produced by ED-Retrieval.
TRAIN_CANDIDATE_ENTITIES=${STORAGE_DATA}/examples/misc/candidate_entities.json
DEV_CANDIDATE_ENTITIES=${STORAGE_DATA}/examples/misc/candidate_entities.json
TEST_CANDIDATE_ENTITIES=${STORAGE_DATA}/examples/misc/candidate_entities.json
ENTITY_DICT=${STORAGE_DATA}/examples/kb/entity_dict.json
# Set the number of demonstrations for LLM-based ED-Reranking. For BLINK-based ED-Reranking, this parameter is not used.
N_DEMONSTRATIONS=1

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

if [ "${METHOD}" == "blink_cross_encoder" ]; then
    python run_ed_reranking_train_eval.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --train_documents ${TRAIN_DOCUMENTS} \
        --dev_documents ${DEV_DOCUMENTS} \
        --test_documents ${TEST_DOCUMENTS} \
        --train_candidate_entities ${TRAIN_CANDIDATE_ENTITIES} \
        --dev_candidate_entities ${DEV_CANDIDATE_ENTITIES} \
        --test_candidate_entities ${TEST_CANDIDATE_ENTITIES} \
        --entity_dict ${ENTITY_DICT} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype train_and_evaluate
fi

if [ "${METHOD}" == "llm_ed" ]; then
    python run_ed_reranking_train_eval.py \
        --method ${METHOD} \
        --config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --train_documents ${TRAIN_DOCUMENTS} \
        --dev_documents ${DEV_DOCUMENTS} \
        --test_documents ${TEST_DOCUMENTS} \
        --train_candidate_entities ${TRAIN_CANDIDATE_ENTITIES} \
        --dev_candidate_entities ${DEV_CANDIDATE_ENTITIES} \
        --test_candidate_entities ${TEST_CANDIDATE_ENTITIES} \
        --entity_dict ${ENTITY_DICT} \
        --n_demonstrations ${N_DEMONSTRATIONS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --actiontype evaluate
fi