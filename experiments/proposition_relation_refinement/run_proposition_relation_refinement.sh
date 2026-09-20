#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/proposition_relation_refinement/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/proposition_relation_refinement/results

######
# Experiment configuration
######

# Method
METHOD=llm_proposition_relation_refiner
CONFIG_PATH=./config/llm_proposition_relation_refiner.conf
CONFIG_NAME=gpt5_4_temporal

# Input Data
INPUT_TRIPLES=${STORAGE_DATA}/examples/misc/triples.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_proposition_relation_refinement.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_triples ${INPUT_TRIPLES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    "$@"
