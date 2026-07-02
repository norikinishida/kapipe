#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/EXPERIMENT_NAME/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/EXPERIMENT_NAME/results

# STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/DATASET_CATEGORY_NAME
# STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/EXPERIMENT_NAME/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
CONFIG_NAME=CONFIG_NAME_A

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

