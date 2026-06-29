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
IDENTIFIER=blink_bi_encoder_cdr

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
    --identifier ${IDENTIFIER} \
    --input_documents ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
