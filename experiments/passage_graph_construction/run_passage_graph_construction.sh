#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/passage_graph_construction/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/passage_graph_construction/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
CONFIG_NAME=default

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/passages.jsonl
INPUT_TRIPLES=${STORAGE_DATA}/examples/triples.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_passage_graph_construction.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_passages ${INPUT_PASSAGES} \
    --input_triples ${INPUT_TRIPLES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
