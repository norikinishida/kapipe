#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/qa/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/qa/results

# STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/qa
# STORAGE_RESULTS=/home/nishida/storage/projects/kapipe/experiments/qa/results

######
# Experiment configuration
######

# Method
METHOD=llm_qa
CONFIG_PATH=./config/llm_qa.conf
CONFIG_NAME=gpt4o_with_context
# CONFIG_NAME=gpt4o_without_context

# Input Data
INPUT_QUESTIONS=${STORAGE_DATA}/examples/questions.json
INPUT_CONTEXTS=${STORAGE_DATA}/examples/questions.contexts.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

GOLD_ANSWERS=${STORAGE_DATA}/examples/questions_with_answers.json

######
# Experiment execution
######

python run_qa.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_questions ${INPUT_QUESTIONS} \
    --input_contexts ${INPUT_CONTEXTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --do_evaluation \
    --gold ${GOLD_ANSWERS}
