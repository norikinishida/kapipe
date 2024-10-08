#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

TRAIN=${STORAGE_DATA}/ed/cdr/train.json
DEV=${STORAGE_DATA}/ed/cdr/dev.json
TEST=${STORAGE_DATA}/ed/cdr/test.json

CAND_ENT_DIR=${STORAGE_RESULTS}/entityretrievalbiencoder/entityretrievalbiencodermodel_scibertuncased_cdr/raiden183a
TRAIN_CAND_ENT=${CAND_ENT_DIR}/train.pred_candidate_entities.json
DEV_CAND_ENT=${CAND_ENT_DIR}/dev.pred_candidate_entities.json
TEST_CAND_ENT=${CAND_ENT_DIR}/test.pred_candidate_entities.json

ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

TRAIN_DEMOS=${STORAGE_DATA}/ed/cdr-demos/train.demonstrations.5.first.json
DEV_DEMOS=${STORAGE_DATA}/ed/cdr-demos/dev.demonstrations.5.first.json
TEST_DEMOS=${STORAGE_DATA}/ed/cdr-demos/test.demonstrations.5.first.json

GPU=0

CONFIG_PATH=./config/llmed.conf

CONFIG_NAME=llm_llama3_8b_cdr_prompt04

python run_llmed.py \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_candidate_entities ${TRAIN_CAND_ENT} \
    --dev_candidate_entities ${DEV_CAND_ENT} \
    --test_candidate_entities ${TEST_CAND_ENT} \
    --entity_dict ${ENTITY_DICT} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --results_dir ${STORAGE_RESULTS} \
    --gpu ${GPU} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate



