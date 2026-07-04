#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/rag_pipeline/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/rag_pipeline/results

######
# Experiment configuration
######

# Method
METHOD=default
CONFIG_PATH=./config/default.conf
CONFIG_NAME=qwen3emb0.6b_gpt5.4nano

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/passages.jsonl
INPUT_QUESTIONS=${STORAGE_DATA}/examples/questions.json
GOLD_QUESTIONS=${STORAGE_DATA}/examples/questions_with_answers.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

######
# Experiment execution
######

# python run_rag_pipeline.py \
#     --method ${METHOD} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --input_file ${INPUT_PASSAGES} \
#     --results_dir ${RESULTS_DIR} \
#     --prefix ${MYPREFIX} \
#     --actiontype indexing

python run_rag_pipeline.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_file ${INPUT_QUESTIONS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype inference \
    --do_evaluation \
    --gold_questions ${GOLD_QUESTIONS}
