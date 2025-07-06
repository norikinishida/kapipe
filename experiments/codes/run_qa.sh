#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
METHOD=llmqa
CONFIG_PATH=./config/llmqa.conf
CONFIG_NAME=openai_gpt4o_cdrqa_prompt02_with_context

# Input Data
DEV_QUESTIONS=${STORAGE_DATA}/examples/questions.json
TEST_QUESTIONS=${STORAGE_DATA}/examples/questions.json
DEV_CONTEXTS=${STORAGE_DATA}/examples/questions.contexts.json
TEST_CONTEXTS=${STORAGE_DATA}/examples/questions.contexts.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_qa.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --dev_questions ${DEV_QUESTIONS} \
    --test_questions ${TEST_QUESTIONS} \
    --dev_contexts ${DEV_CONTEXTS} \
    --test_contexts ${TEST_CONTEXTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype inference_only

    # --actiontype evaluate

