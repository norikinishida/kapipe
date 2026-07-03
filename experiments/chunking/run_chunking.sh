#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/chunking/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/chunking/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=config/default.conf
CONFIG_NAME=en_core_sci_md_w100

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/articles.jsonl

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_chunking.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_passages ${INPUT_PASSAGES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

