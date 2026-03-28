#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
METHOD=METHOD_NAME_A

# Input Data
INPUT_SOMETHING=${STORAGE_DATA}/examples/SOMETHING

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_TEMPLATE.py \
    --method ${METHOD} \
    --input_SOMETHING ${INPUT_SOMETHING} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

