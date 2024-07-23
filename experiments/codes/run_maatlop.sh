#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# TRAIN=${STORAGE_DATA}/docre/cdr/train.json
# DEV=${STORAGE_DATA}/docre/cdr/dev.json
# TEST=${STORAGE_DATA}/docre/cdr/test.json
# ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json
#
TRAIN=${STORAGE_DATA}/docre/hoip-v1/train.json
DEV=${STORAGE_DATA}/docre/hoip-v1/dev.json
TEST=${STORAGE_DATA}/docre/hoip-v1/test.json
ENTITY_DICT=${STORAGE_DATA}/kb/hoip-kb/hoip.entity_dict.json

GPU=0

CONFIG_PATH=./config/maatlop.conf

# CONFIG_NAME=maatlopmodel_scibertcased_cdr
# CONFIG_NAME=maatlopmodel_scibertcased_cdr_mc
CONFIG_NAME=maatlopmodel_scibertcased_hoip
# CONFIG_NAME=maatlopmodel_scibertcased_hoip_mc

python run_maatlop.py \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --gpu ${GPU} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

