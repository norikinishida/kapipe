#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
SPACY_MODEL_NAME=en_core_sci_md
WINDOW_SIZE=100

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/reports.jsonl

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_chunking.py \
    --spacy_model_name ${SPACY_MODEL_NAME} \
    --window_size ${WINDOW_SIZE} \
    --input_passages ${INPUT_PASSAGES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

