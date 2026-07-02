#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/docre/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/docre/results

# STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/docre
# STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/docre/results

######
# Experiment configuration
######

# Method
# METHOD=atlop
METHOD=llm_docre

if [ "${METHOD}" == "atlop" ]; then
    IDENTIFIER=atlop_cdr
elif [ "${METHOD}" == "llm_docre" ]; then
    IDENTIFIER=llm_docre_cdr
    LLM_PROVIDER=openai
    LLM_MODEL_NAME=gpt-5.4-nano
    LLM_MAX_NEW_TOKENS=1024
else
    echo "Error: Invalid METHOD specified."
    exit 1
fi

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents.ner.ed_ret.ed_rank.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

if [ "${METHOD}" == "atlop" ]; then
    python run_docre.py \
        --method ${METHOD} \
        --identifier ${IDENTIFIER} \
        --input_documents ${DOCUMENTS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX}
fi

if [ "${METHOD}" == "llm_docre" ]; then
    python run_docre.py \
        --method ${METHOD} \
        --identifier ${IDENTIFIER} \
        --input_documents ${DOCUMENTS} \
        --results_dir ${RESULTS_DIR} \
        --prefix ${MYPREFIX} \
        --llm_provider ${LLM_PROVIDER} \
        --llm_model_name ${LLM_MODEL_NAME} \
        --llm_max_new_tokens ${LLM_MAX_NEW_TOKENS}
fi
