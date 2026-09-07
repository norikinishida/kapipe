#!/usr/bin/env bash

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/passage_retrieval/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/passage_retrieval/results

######
# Experiment configuration
######

# Method
# METHOD=bm25
# METHOD=contriever
METHOD=qwen3_embedding

if [ "${METHOD}" = "bm25" ]; then
    CONFIG_PATH=./config/bm25.conf
    CONFIG_NAME=bm25_top3
elif [ "${METHOD}" = "contriever" ]; then
    CONFIG_PATH=./config/contriever.conf
    CONFIG_NAME=contriever_msmarco_top3
elif [ "${METHOD}" = "qwen3_embedding" ]; then
    CONFIG_PATH=./config/qwen3_embedding.conf
    CONFIG_NAME=qwen3_embedding_06b_top3
else
    echo "Unknown method: ${METHOD}"
    exit 1
fi

# Input Data
INPUT_PASSAGES=${STORAGE_DATA}/examples/passages.jsonl
INPUT_QUESTIONS=${STORAGE_DATA}/examples/questions.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (optional) Evaluation
GOLD_CONTEXTS=${STORAGE_DATA}/examples/questions.gold_contexts.json

######
# Experiment execution
######

python run_passage_retrieval.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_file ${INPUT_PASSAGES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype indexing

python run_passage_retrieval.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_file ${INPUT_QUESTIONS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype search \
    --do_evaluation \
    --gold ${GOLD_CONTEXTS}