#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/context_formatting/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/context_formatting/results

######
# Experiment configuration
######

# Method
METHOD=graph_verbalizer
CONFIG_PATH=./config/graph_verbalizer.conf
CONFIG_NAME=temporal

# Input Data
INPUT_GRAPH_CONTEXTS=${STORAGE_DATA}/examples/questions.contexts.graph_retrieval.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_context_formatting.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_graph_contexts ${INPUT_GRAPH_CONTEXTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
