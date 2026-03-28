#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
IDENTIFIER=biaffine_ner_cdr
# IDENTIFIER=gpt4omini_cdr
# IDENTIFIER=qwen2_5_7b_cdr
# IDENTIFIER=gpt4omini_any

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_ner.py \
    --identifier ${IDENTIFIER} \
    --input_documents ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
