#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
# (BLINK Cross-Encoder)
METHOD=blinkcrossencoder
CONFIG_PATH=./config/blinkcrossencoder.conf
CONFIG_NAME=blinkcrossencodermodel_scibertuncased_cdr
# (LLM-ED)
# METHOD=llmed
# CONFIG_PATH=./config/llmed.conf
# CONFIG_NAME=openai_gpt4omini_cdr_prompt09fewshot

# Input Data
TRAIN_DOCS=${STORAGE_DATA}/ed/cdr/train.json
DEV_DOCS=${STORAGE_DATA}/ed/cdr/dev.json
TEST_DOCS=${STORAGE_DATA}/ed/cdr/test.json
#
CANDS_ROOT=${STORAGE_RESULTS}/edret/blinkbiencoder/blinkbiencodermodel_scibertuncased_cdr/raiden512a
TRAIN_CANDS=${CANDS_ROOT}/train.pred_candidate_entities.json
DEV_CANDS=${CANDS_ROOT}/dev.pred_candidate_entities.json
TEST_CANDS=${CANDS_ROOT}/test.pred_candidate_entities.json
ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json
#
TRAIN_DEMOS=${STORAGE_DATA}/ed/cdr-demos/train.demonstrations.5.count.json
DEV_DEMOS=${STORAGE_DATA}/ed/cdr-demos/dev.demonstrations.5.count.json
TEST_DEMOS=${STORAGE_DATA}/ed/cdr-demos/test.demonstrations.5.count.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (BLINK Cross-Encoder)
python run_edrank.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --train_documents ${TRAIN_DOCS} \
    --dev_documents ${DEV_DOCS} \
    --test_documents ${TEST_DOCS} \
    --train_candidate_entities ${TRAIN_CANDS} \
    --dev_candidate_entities ${DEV_CANDS} \
    --test_candidate_entities ${TEST_CANDS} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype train_and_evaluate

# (LLM-ED)
# python run_edrank.py \
#     --method ${METHOD} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --train_documents ${TRAIN_DOCS} \
#     --dev_documents ${DEV_DOCS} \
#     --test_documents ${TEST_DOCS} \
#     --train_candidate_entities ${TRAIN_CANDS} \
#     --dev_candidate_entities ${DEV_CANDS} \
#     --test_candidate_entities ${TEST_CANDS} \
#     --train_demonstrations ${TRAIN_DEMOS} \
#     --dev_demonstrations ${DEV_DEMOS} \
#     --test_demonstrations ${TEST_DEMOS} \
#     --entity_dict ${ENTITY_DICT} \
#     --results_dir ${RESULTS_DIR} \
#     --prefix ${MYPREFIX} \
#     --actiontype evaluate

