#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/proposition_extraction/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/proposition_extraction/results

######
# Experiment configuration
######

# Method
METHOD=llm_proposition_extractor
CONFIG_PATH=./config/llm_proposition_extractor.conf
CONFIG_NAME=gpt5_4_nano

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/articles.jsonl

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_proposition_extraction.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_passages ${INPUT_PASSAGES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
