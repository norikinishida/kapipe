#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Input Data
# DOCUMENTS=${STORAGE_DATA}/examples/documents.json
DOCUMENTS=${STORAGE_DATA}/examples/documents_with_triples.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example3

python run_triple_extraction_pipeline.py \
    --input_documents ${DOCUMENTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

    # --do_evaluation

