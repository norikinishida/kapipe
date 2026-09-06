#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/graph_retrieval/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/graph_retrieval/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
CONFIG_NAME=hop1_temporal

# Input Data
INPUT_GRAPH=${STORAGE_DATA}/examples/graph.graphml
INPUT_ANCHOR_CONTEXTS=${STORAGE_DATA}/examples/questions.contexts.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_graph_retrieval.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_graph ${INPUT_GRAPH} \
    --input_anchor_contexts ${INPUT_ANCHOR_CONTEXTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
