#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/ner/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/ner/results

# STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/ner
# STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/ner/results

######
# Experiment configuration
######

# Method
# METHOD=biaffine_ner
METHOD=llm_ner

if [ "${METHOD}" == "biaffine_ner" ]; then
    IDENTIFIER=biaffine_ner_cdr
elif [ "${METHOD}" == "llm_ner" ]; then
    IDENTIFIER=llm_ner_cdr
    LLM_PROVIDER=openai
    LLM_MODEL_NAME=gpt-5.4-nano
    LLM_MAX_NEW_TOKENS=1024
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

if [ "${METHOD}" == "biaffine_ner" ]; then
    python run_ner.py \
        --method ${METHOD} \
        --identifier ${IDENTIFIER} \
        --input_documents ${DOCUMENTS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX}
fi

if [ "${METHOD}" == "llm_ner" ]; then
    python run_ner.py \
        --method ${METHOD} \
        --identifier ${IDENTIFIER} \
        --input_documents ${DOCUMENTS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --llm_provider ${LLM_PROVIDER} \
        --llm_model_name ${LLM_MODEL_NAME} \
        --llm_max_new_tokens ${LLM_MAX_NEW_TOKENS}
fi
