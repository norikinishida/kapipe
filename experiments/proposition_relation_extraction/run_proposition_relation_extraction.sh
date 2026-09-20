#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/proposition_relation_extraction/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/proposition_relation_extraction/results

######
# Experiment configuration
######

# Method
METHOD=llm_proposition_relation_extractor
CONFIG_PATH=./config/llm_proposition_relation_extractor.conf
CONFIG_NAME=gpt5_4_nano_contriever_temporal

# Input Data
INPUT_PROPOSITIONS=${STORAGE_DATA}/examples/misc/propositions.jsonl

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_proposition_relation_extraction.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_propositions ${INPUT_PROPOSITIONS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    "$@"
