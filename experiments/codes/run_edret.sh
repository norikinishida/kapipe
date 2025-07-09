#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

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
# (In practice, use separate files for training, validation, and test data.
# This example checks whether the model can achieve near 100% accuracy (i.e., overfit) on the training data.)
TRAIN_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
DEV_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
TEST_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
ENTITY_DICT=${STORAGE_DATA}/examples/entity_dict.json

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
