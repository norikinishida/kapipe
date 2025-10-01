#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
IDENTIFIER=gpt4o_with_context

# Input Data
QUESTIONS=${STORAGE_DATA}/examples/questions.json
CONTEXTS=${STORAGE_DATA}/examples/questions.contexts.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_qa.py \
    --identifier ${IDENTIFIER} \
    --input_questions ${QUESTIONS} \
    --input_contexts ${CONTEXTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
