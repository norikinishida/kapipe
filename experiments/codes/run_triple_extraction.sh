#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
IDENTIFIER=biaffinener_blink_blink_atlop_cdr
# IDENTIFIER=llmner_blink_llmed_llmdocre_cdr
# IDENTIFIER=biaffinener_blink_blink_atlop_linked_docred
# IDENTIFIER=llmner_blink_llmed_llmdocre_linked_docred

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents_without_triples.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_triple_extraction.py \
    --identifier ${IDENTIFIER} \
    --input_documents_list ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

