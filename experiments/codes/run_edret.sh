#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

################
# Method: BLINK Bi-Encoder
# Dataset: CDR
################

TRAIN=${STORAGE_DATA}/ed/cdr/train.json
DEV=${STORAGE_DATA}/ed/cdr/dev.json
TEST=${STORAGE_DATA}/ed/cdr/test.json

ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

CONFIG_PATH=./config/blinkbiencoder.conf
CONFIG_NAME=blinkbiencodermodel_scibertuncased_cdr

python run_edret.py \
    --gpu 0 \
    --method blinkbiencoder \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

################
# Method: BLINK Bi-Encoder
# Dataset: Linked-DocRED
################

TRAIN=${STORAGE_DATA}/ed/linked-docred/train.json
DEV=${STORAGE_DATA}/ed/linked-docred/dev.json
TEST=${STORAGE_DATA}/ed/linked-docred/test.json

ENTITY_DICT=${STORAGE_DATA}/kb/dbpedia/dbpedia20200201.entity_dict.json

CONFIG_PATH=./config/blinkbiencoder.conf
CONFIG_NAME=blinkbiencodermodel_bertbaseuncased_linked_docred

python run_edret.py \
    --gpu 0 \
    --method blinkbiencoder \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

################
# Method: BLINK Bi-Encoder
# Dataset: MedMentions
################

TRAIN=${STORAGE_DATA}/ed/medmentions/train.json
DEV=${STORAGE_DATA}/ed/medmentions/dev.json
TEST=${STORAGE_DATA}/ed/medmentions/test.json

ENTITY_DICT=${STORAGE_DATA}/kb/umls/umls2017aa.entity_dict.json

CONFIG_PATH=./config/blinkbiencoder.conf
CONFIG_NAME=blinkbiencodermodel_scibertuncased_medmentions

python run_edret.py \
    --gpu 0 \
    --method blinkbiencoder \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

################
# Method: LexicalEntityRetriever (BM25 or Levenshtein)
# Dataset: CDR
################

TRAIN=${STORAGE_DATA}/ed/cdr/train.json
DEV=${STORAGE_DATA}/ed/cdr/dev.json
TEST=${STORAGE_DATA}/ed/cdr/test.json

ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

CONFIG_PATH=./config/lexicalentityretriever.conf
CONFIG_NAME=bm25_cdr
# CONFIG_NAME=bm25_with_desc_cdr
# CONFIG_NAME=levenshtein_cdr

python run_edret.py \
    --gpu 0 \
    --method lexicalentityretriever \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate

