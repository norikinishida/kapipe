#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
# (BLINK Bi-Encoder)
METHOD=blinkbiencoder
CONFIG_PATH=./config/blinkbiencoder.conf
CONFIG_NAME=blinkbiencodermodel_scibertuncased_cdr
# (BM25)
# METHOD=lexicalentityretriever
# CONFIG_PATH=./config/lexicalentityretriever.conf
# CONFIG_NAME=bm25_cdr

# Input Data
TRAIN_DOCS=${STORAGE_DATA}/ed/cdr/train.json
DEV_DOCS=${STORAGE_DATA}/ed/cdr/dev.json
TEST_DOCS=${STORAGE_DATA}/ed/cdr/test.json
ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (BLINK Bi-Encoder)
python run_edret.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --train_documents ${TRAIN_DOCS} \
    --dev_documents ${DEV_DOCS} \
    --test_documents ${TEST_DOCS} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype train_and_evaluate
# (BM25)
# python run_edret.py \
#     --method ${METHOD} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --train_documents ${TRAIN_DOCS} \
#     --dev_documents ${DEV_DOCS} \
#     --test_documents ${TEST_DOCS} \
#     --entity_dict ${ENTITY_DICT} \
#     --results_dir ${RESULTS_DIR} \
#     --prefix ${MYPREFIX} \
#     --actiontype evaluate
