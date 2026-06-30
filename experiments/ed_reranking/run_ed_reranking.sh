#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ed_reranking/data
# STORAGE_DATA=/home/nishida/projects/kapipe/experiments/datasets/ed
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ed_reranking/results

######
# Experiment configuration
######

# Method
# METHOD=blink_cross_encoder
METHOD=llm_ed

if [ "${METHOD}" == "blink_cross_encoder" ]; then
    IDENTIFIER=blink_cross_encoder_cdr
elif [ "${METHOD}" == "llm_ed" ]; then
    IDENTIFIER=llm_ed_cdr
    LLM_PROVIDER=openai
    LLM_MODEL_NAME=gpt-5.4-nano
    LLM_MAX_NEW_TOKENS=1024
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents.ner.ed_ret.json
CANDIDATE_ENTITIES=${STORAGE_DATA}/examples/candidate_entities.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_ed_reranking.py \
    --method ${METHOD} \
    --identifier ${IDENTIFIER} \
    --input_documents ${DOCUMENTS} \
    --input_candidate_entities ${CANDIDATE_ENTITIES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

