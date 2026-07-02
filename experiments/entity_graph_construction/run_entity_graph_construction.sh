#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/entity_graph_construction/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/entity_graph_construction/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
CONFIG_NAME=keep

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents_with_triples.json
ADDITIONAL_TRIPLES=${STORAGE_DATA}/examples/additional_triples.json
ENTITY_DICT=${STORAGE_DATA}/examples/entity_dict.json

# Output Data
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_entity_graph_construction.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents_list ${DOCUMENTS} \
    --input_additional_triples ${ADDITIONAL_TRIPLES} \
    --input_entity_dict ${ENTITY_DICT} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
