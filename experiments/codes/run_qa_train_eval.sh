#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
METHOD=llm_qa
CONFIG_PATH=./config/llm_qa.conf
CONFIG_NAME=openai_gpt4o_cdrqa_prompt03_with_context
# CONFIG_NAME=hf_llama3_1_8b_cdrqa_prompt03_with_context
# CONFIG_NAME=hf_llama3_1_70b_cdrqa_prompt03_with_context
# CONFIG_NAME=hf_qwen2_5_7b_cdrqa_prompt03_with_context

# Input Data
# (In practice, use separate files for training, validation, and test data.)
DEV_QUESTIONS=${STORAGE_DATA}/examples/questions_with_answers.json
TEST_QUESTIONS=${STORAGE_DATA}/examples/questions_with_answers.json
DEV_CONTEXTS=${STORAGE_DATA}/examples/questions.contexts.json
TEST_CONTEXTS=${STORAGE_DATA}/examples/questions.contexts.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_qa_train_eval.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --dev_questions ${DEV_QUESTIONS} \
    --test_questions ${TEST_QUESTIONS} \
    --dev_contexts ${DEV_CONTEXTS} \
    --test_contexts ${TEST_CONTEXTS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype evaluate

