#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ed_retrieval/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ed_retrieval/results

######
# Experiment configuration
######

# Method
# METHOD=mention_name_entity_retriever
METHOD=blink_bi_encoder

if [ "${METHOD}" == "mention_name_entity_retriever" ]; then
    CONFIG_PATH=./config/mention_name_entity_retriever.conf
    CONFIG_NAME=default
elif [ "${METHOD}" == "blink_bi_encoder" ]; then
    CONFIG_PATH=./config/blink_bi_encoder.conf
    CONFIG_NAME=blink_bi_encoder_cdr
    # CONFIG_NAME=blink_bi_encoder_linked_docred
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
INPUT_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_typed_mentions.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_DOCUMENTS=${STORAGE_DATA}/examples/ed/documents_with_disambiguated_entities.json

######
# Experiment execution
######

python run_ed_retrieval.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_documents ${INPUT_DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --do_evaluation \
    --gold ${GOLD_DOCUMENTS}
