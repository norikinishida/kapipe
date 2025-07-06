#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

################
# Method: BLINK Cross-Encoder
# Dataset: CDR
################

TRAIN=${STORAGE_DATA}/ed/cdr/train.json
DEV=${STORAGE_DATA}/ed/cdr/dev.json
TEST=${STORAGE_DATA}/ed/cdr/test.json

# CAND_ENT_DIR=${STORAGE_RESULTS}/edret/blinkbiencoder/blinkbiencodermodel_scibertuncased_cdr/raiden183a
CAND_ENT_DIR=${STORAGE_RESULTS}/edret/blinkbiencoder/blinkbiencodermodel_scibertuncased_cdr/raiden512a
TRAIN_CAND_ENT=${CAND_ENT_DIR}/train.pred_candidate_entities.json
DEV_CAND_ENT=${CAND_ENT_DIR}/dev.pred_candidate_entities.json
TEST_CAND_ENT=${CAND_ENT_DIR}/test.pred_candidate_entities.json

ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

CONFIG_PATH=./config/blinkcrossencoder.conf
CONFIG_NAME=blinkcrossencodermodel_scibertuncased_cdr

python run_edrank.py \
    --gpu 0 \
    --method blinkcrossencoder \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_candidate_entities ${TRAIN_CAND_ENT} \
    --dev_candidate_entities ${DEV_CAND_ENT} \
    --test_candidate_entities ${TEST_CAND_ENT} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

################
# Method: BLINK Cross-Encoder
# Dataset: Linked-DocRED
################

TRAIN=${STORAGE_DATA}/ed/linked-docred/train.json
DEV=${STORAGE_DATA}/ed/linked-docred/dev.json
TEST=${STORAGE_DATA}/ed/linked-docred/test.json

CAND_ENT_DIR=${STORAGE_RESULTS}/edret/blinkbiencoder/blinkbiencodermodel_bertbaseuncased_linked_docred/raiden906a
TRAIN_CAND_ENT=${CAND_ENT_DIR}/train.pred_candidate_entities.json
DEV_CAND_ENT=${CAND_ENT_DIR}/dev.pred_candidate_entities.json
TEST_CAND_ENT=${CAND_ENT_DIR}/test.pred_candidate_entities.json

ENTITY_DICT=${STORAGE_DATA}/kb/dbpedia/dbpedia20200201.entity_dict.json

CONFIG_PATH=./config/blinkcrossencoder.conf
CONFIG_NAME=blinkcrossencodermodel_bertbaseuncased_linked_docred

python run_edrank.py \
    --gpu 0 \
    --method blinkcrossencoder \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_candidate_entities ${TRAIN_CAND_ENT} \
    --dev_candidate_entities ${DEV_CAND_ENT} \
    --test_candidate_entities ${TEST_CAND_ENT} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

################
# Method: BLINK Cross-Encoder
# Dataset: MedMentions
################

TRAIN=${STORAGE_DATA}/ed/medmentions/train.json
DEV=${STORAGE_DATA}/ed/medmentions/dev.json
TEST=${STORAGE_DATA}/ed/medmentions/test.json

CAND_ENT_DIR=${STORAGE_RESULTS}/edret/blinkbiencoder/blinkbiencodermodel_scibertuncased_medmentions/raiden509a
TRAIN_CAND_ENT=${CAND_ENT_DIR}/train.pred_candidate_entities.json
DEV_CAND_ENT=${CAND_ENT_DIR}/dev.pred_candidate_entities.json
TEST_CAND_ENT=${CAND_ENT_DIR}/test.pred_candidate_entities.json

ENTITY_DICT=${STORAGE_DATA}/kb/umls/umls2017aa.entity_dict.json

CONFIG_PATH=./config/blinkcrossencoder.conf
CONFIG_NAME=blinkcrossencodermodel_scibertuncased_medmentions

python run_edrank.py \
    --gpu 0 \
    --method blinkcrossencoder \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_candidate_entities ${TRAIN_CAND_ENT} \
    --dev_candidate_entities ${DEV_CAND_ENT} \
    --test_candidate_entities ${TEST_CAND_ENT} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

################
# Method: LLM-ED
# Dataset: CDR
################

TRAIN=${STORAGE_DATA}/ed/cdr/train.json
DEV=${STORAGE_DATA}/ed/cdr/dev.json
TEST=${STORAGE_DATA}/ed/cdr/test.json

CAND_ENT_DIR=${STORAGE_RESULTS}/edret/blinkbiencoder/blinkbiencodermodel_scibertuncased_cdr/raiden512a
TRAIN_CAND_ENT=${CAND_ENT_DIR}/train.pred_candidate_entities.json
DEV_CAND_ENT=${CAND_ENT_DIR}/dev.pred_candidate_entities.json
TEST_CAND_ENT=${CAND_ENT_DIR}/test.pred_candidate_entities.json

TRAIN_DEMOS=${STORAGE_DATA}/ed/cdr-demos/train.demonstrations.5.count.json
DEV_DEMOS=${STORAGE_DATA}/ed/cdr-demos/dev.demonstrations.5.count.json
TEST_DEMOS=${STORAGE_DATA}/ed/cdr-demos/test.demonstrations.5.count.json

ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

CONFIG_PATH=./config/llmed.conf
CONFIG_NAME=openai_gpt4omini_cdr_prompt09fewshot

python run_edrank.py \
    --gpu 0 \
    --method llmed \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_candidate_entities ${TRAIN_CAND_ENT} \
    --dev_candidate_entities ${DEV_CAND_ENT} \
    --test_candidate_entities ${TEST_CAND_ENT} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate

################
# Method: LLM-ED
# Dataset: Linked-DocRED
################

TRAIN=${STORAGE_DATA}/ed/linked-docred/train.json
DEV=${STORAGE_DATA}/ed/linked-docred/dev.json
TEST=${STORAGE_DATA}/ed/linked-docred/test.json

CAND_ENT_DIR=${STORAGE_RESULTS}/edret/blinkbiencoder/blinkbiencodermodel_bertbaseuncased_linked_docred/raiden906a
TRAIN_CAND_ENT=${CAND_ENT_DIR}/train.pred_candidate_entities.json
DEV_CAND_ENT=${CAND_ENT_DIR}/dev.pred_candidate_entities.json
TEST_CAND_ENT=${CAND_ENT_DIR}/test.pred_candidate_entities.json

TRAIN_DEMOS=${STORAGE_DATA}/ed/linked-docred-demos/train.demonstrations.5.count.json
DEV_DEMOS=${STORAGE_DATA}/ed/linked-docred-demos/dev.demonstrations.5.count.json
TEST_DEMOS=${STORAGE_DATA}/ed/linked-docred-demos/test.demonstrations.5.count.json

ENTITY_DICT=${STORAGE_DATA}/kb/dbpedia/dbpedia20200201.entity_dict.json

CONFIG_PATH=./config/llmed.conf
CONFIG_NAME=openai_gpt4omini_linked_docred_prompt09fewshot

python run_edrank.py \
    --gpu 0 \
    --method llmed \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_candidate_entities ${TRAIN_CAND_ENT} \
    --dev_candidate_entities ${DEV_CAND_ENT} \
    --test_candidate_entities ${TEST_CAND_ENT} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate

