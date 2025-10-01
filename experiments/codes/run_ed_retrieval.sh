#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
IDENTIFIER=blinkbiencoder_cdr

# Input Data
DOCUMENTS=${STORAGE_RESULTS}/ner/ner/biaffinener_cdr/example/documents.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_ed_retrieval.py \
    --identifier ${IDENTIFIER} \
    --input_documents ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
