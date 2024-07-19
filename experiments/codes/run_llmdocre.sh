#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

TRAIN=${STORAGE_DATA}/docre/cdr/train.json
DEV=${STORAGE_DATA}/docre/cdr/dev.json
TEST=${STORAGE_DATA}/docre/cdr/test.json
ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json
TRAIN_DEMOS=${STORAGE_DATA}/docre/cdr-demos/train.demonstrations.5.first_adjusted.json
DEV_DEMOS=${STORAGE_DATA}/docre/cdr-demos/dev.demonstrations.5.first_adjusted.json
TEST_DEMOS=${STORAGE_DATA}/docre/cdr-demos/test.demonstrations.5.first_adjusted.json
#
# TRAIN=${STORAGE_DATA}/docre/hoip-v5/train.merged.json
# DEV=${STORAGE_DATA}/docre/hoip-v5/dev.merged.json
# TEST=${STORAGE_DATA}/docre/hoip-v5/test.merged.json
# ENTITY_DICT=${STORAGE_DATA}/kb/hoip-kb/hoip.entity_dict.json
# TRAIN_DEMOS=${STORAGE_DATA}/docre/hoip-v5-demos/train.demonstrations.5.random.json
# DEV_DEMOS=${STORAGE_DATA}/docre/hoip-v5-demos/dev.demonstrations.5.random.json
# TEST_DEMOS=${STORAGE_DATA}/docre/hoip-v5-demos/test.demonstrations.5.random.json

GPU=0

CONFIG_PATH=./config/llmdocre.conf

CONFIG_NAME=llm_llama3_8b_cdr_prompt03
# CONFIG_NAME=llm_llama3_8b_hoip_prompt03_cn

python run_llmdocre.py \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --entity_dict ${ENTITY_DICT} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --results_dir ${STORAGE_RESULTS} \
    --gpu ${GPU} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate


