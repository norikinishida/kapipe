#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

TRAIN=${STORAGE_DATA}/ed/cdr/train.json
DEV=${STORAGE_DATA}/ed/cdr/dev.json
TEST=${STORAGE_DATA}/ed/cdr/test.json

ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

GPU=0

CONFIG_PATH=./config/lexicalentityretrieval.conf

CONFIG_NAME=bm25_cdr
# CONFIG_NAME=bm25_with_desc_cdr
# CONFIG_NAME=levenshtein_cdr

python run_lexicalentityretrieval.py \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate

