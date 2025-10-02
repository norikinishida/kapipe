#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
IDENTIFIER=atlop_cdr
# IDENTIFIER=gpt4omini_cdr
# IDENTIFIER=qwen2_5_7b_cdr

# Input Data
DOCUMENTS=${STORAGE_RESULTS}/ed_reranking/ed_reranking/blink_cross_encoder_cdr/example/documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_docre.py \
    --identifier ${IDENTIFIER} \
    --input_documents ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

