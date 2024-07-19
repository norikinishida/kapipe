#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

TRAIN=${STORAGE_DATA}/ner/cdr/train.json
DEV=${STORAGE_DATA}/ner/cdr/dev.json
TEST=${STORAGE_DATA}/ner/cdr/test.json
TRAIN_DEMOS=${STORAGE_DATA}/ner/cdr-demos/train.demonstrations.5.first.json
DEV_DEMOS=${STORAGE_DATA}/ner/cdr-demos/dev.demonstrations.5.first.json
TEST_DEMOS=${STORAGE_DATA}/ner/cdr-demos/test.demonstrations.5.first.json
#
# TRAIN=${STORAGE_DATA}/ner/conll2003/train.json
# DEV=${STORAGE_DATA}/ner/conll2003/testa.json
# TEST=${STORAGE_DATA}/ner/conll2003/testb.json
# TRAIN_DEMOS=${STORAGE_DATA}/ner/conll2003-demos/train.demonstrations.5.first.json
# DEV_DEMOS=${STORAGE_DATA}/ner/conll2003-demos/dev.demonstrations.5.first.json
# TEST_DEMOS=${STORAGE_DATA}/ner/conll2003-demos/test.demonstrations.5.first.json

GPU=0

CONFIG_PATH=./config/llmner.conf

CONFIG_NAME=llm_llama3_8b_cdr_prompt07

python run_llmner.py \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --results_dir ${STORAGE_RESULTS} \
    --gpu ${GPU} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate

