#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
# IDENTIFIER=blinkcrossencoder_cdr
IDENTIFIER=gpt4omini_cdr
# IDENTIFIER=qwen2_5_7b_cdr

# Input Data
DOCUMENTS=${STORAGE_RESULTS}/ed_retrieval/ed_retrieval/blinkbiencoder_cdr/example/documents.json
CANDS=${STORAGE_RESULTS}/ed_retrieval/ed_retrieval/blinkbiencoder_cdr/example/candidate_entities.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_ed_reranking.py \
    --identifier ${IDENTIFIER} \
    --input_documents ${DOCUMENTS} \
    --input_candidate_entities ${CANDS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

