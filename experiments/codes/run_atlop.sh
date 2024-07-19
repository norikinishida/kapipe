#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

TRAIN=${STORAGE_DATA}/docre/cdr/train.json
DEV=${STORAGE_DATA}/docre/cdr/dev.json
TEST=${STORAGE_DATA}/docre/cdr/test.json
#
# TRAIN=${STORAGE_DATA}/docre/gda/train.json
# DEV=${STORAGE_DATA}/docre/gda/dev.json
# TEST=${STORAGE_DATA}/docre/gda/test.json
#
# TRAIN=${STORAGE_DATA}/docre/docred/train.json
# DEV=${STORAGE_DATA}/docre/docred/dev.json
# TEST=${STORAGE_DATA}/docre/docred/test.json
#
# TRAIN=${STORAGE_DATA}/docre/redocred/train.json
# DEV=${STORAGE_DATA}/docre/redocred/dev.json
# TEST=${STORAGE_DATA}/docre/redocred/test.json

GPU=0

CONFIG_PATH=./config/atlop.conf

# CONFIG_NAME=atlopmodel_scibertcased_cdr
CONFIG_NAME=atlopmodel_scibertcased_cdr_overlap
# CONFIG_NAME=atlopmodel_scibertcased_gda
# CONFIG_NAME=atlopmodel_bertbasecased_docred
# CONFIG_NAME=atlopmodel_bertbasecased_redocred

python run_atlop.py \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --results_dir ${STORAGE_RESULTS} \
    --gpu ${GPU} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate



