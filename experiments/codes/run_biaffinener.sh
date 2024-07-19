#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

TRAIN=${STORAGE_DATA}/ner/cdr/train.json
DEV=${STORAGE_DATA}/ner/cdr/dev.json
TEST=${STORAGE_DATA}/ner/cdr/test.json
#
# TRAIN=${STORAGE_DATA}/ner/conll2003/train.json
# DEV=${STORAGE_DATA}/ner/conll2003/testa.json
# TEST=${STORAGE_DATA}/ner/conll2003/testb.json

GPU=0

CONFIG_PATH=./config/biaffinener.conf

CONFIG_NAME=biaffinenermodel_scibertuncased_cdr
# CONFIG_NAME=biaffinenermodel_robertalarge_conll2003

python run_biaffinener.py \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --results_dir ${STORAGE_RESULTS} \
    --gpu ${GPU} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

