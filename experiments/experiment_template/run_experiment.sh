#!/usr/bin/env bash

######
# Storage paths
######

EXPERIMENT_NAME="EXPERIMENT_PROJECT_NAME"
DATASET_CATEGORY_NAME="DATASET_CATEGORY_NAME"

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/${EXPERIMENT_NAME}/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/${EXPERIMENT_NAME}/results

# STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/${DATASET_CATEGORY_NAME}
# STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/${EXPERIMENT_NAME}/results

######
# Experiment configuration
######

# Method
METHOD=METHOD_NAME_A
# METHOD=METHOD_NAME_B

# Input Data
INPUT_SOMETHING=${STORAGE_DATA}/SOMETHING

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

python run_experiment.py \
    --method ${METHOD} \
    --input_SOMETHING ${INPUT_SOMETHING} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

