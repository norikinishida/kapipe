#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ed_retrieval/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ed_retrieval/results

# STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/ed
# STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/ed_retrieval/results

######
# Experiment configuration
######

# Method
METHOD=blink_bi_encoder
CONFIG_PATH=./config/blink_bi_encoder.conf
CONFIG_NAME=blink_bi_encoder_cdr

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents.ner.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_ed_retrieval.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
