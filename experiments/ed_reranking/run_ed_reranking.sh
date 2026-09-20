#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ed_reranking/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ed_reranking/results

######
# Experiment configuration
######

# Method
# METHOD=identical_entity_reranker
# METHOD=blink_cross_encoder
METHOD=llm_ed

if [ "${METHOD}" == "identical_entity_reranker" ]; then
    CONFIG_PATH=./config/identical_entity_reranker.conf
    CONFIG_NAME=default
elif [ "${METHOD}" == "blink_cross_encoder" ]; then
    CONFIG_PATH=./config/blink_cross_encoder.conf
    CONFIG_NAME=blink_cross_encoder_cdr
elif [ "${METHOD}" == "llm_ed" ]; then
    CONFIG_PATH=./config/llm_ed.conf
    CONFIG_NAME=llm_ed_cdr
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
INPUT_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json
CANDIDATE_ENTITIES=${STORAGE_DATA}/examples/misc/candidate_entities.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# Evaluation
GOLD_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json

######
# Experiment execution
######

python run_ed_reranking.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents ${INPUT_DOCUMENTS} \
    --input_candidate_entities ${CANDIDATE_ENTITIES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --do_evaluation \
    --gold ${GOLD_DOCUMENTS} \
    "$@"

