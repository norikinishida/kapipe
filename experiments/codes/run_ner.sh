#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

##############
# Method: Biaffine-NER
# Dataset: CDR
##############

TRAIN=${STORAGE_DATA}/ner/cdr/train.json
DEV=${STORAGE_DATA}/ner/cdr/dev.json
TEST=${STORAGE_DATA}/ner/cdr/test.json

DATASET_NAME=cdr

CONFIG_PATH=./config/biaffinener.conf
CONFIG_NAME=biaffinenermodel_scibertuncased_cdr

python run_ner.py \
    --gpu 0 \
    --method biaffinener \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --dataset_name ${DATASET_NAME} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

##############
# Method: Biaffine-NER
# Dataset: Linked-DocRED
##############

TRAIN=${STORAGE_DATA}/ner/linked-docred/train.json
DEV=${STORAGE_DATA}/ner/linked-docred/dev.json
TEST=${STORAGE_DATA}/ner/linked-docred/test.json

DATASET_NAME=linked_docred

CONFIG_PATH=./config/biaffinener.conf
CONFIG_NAME=biaffinenermodel_bertbaseuncased_linked_docred

python run_ner.py \
    --gpu 0 \
    --method biaffinener \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --dataset_name ${DATASET_NAME} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

##############
# Method: Biaffine-NER
# Dataset: MedMentions
##############

TRAIN=${STORAGE_DATA}/ner/medmentions/train.json
DEV=${STORAGE_DATA}/ner/medmentions/dev.json
TEST=${STORAGE_DATA}/ner/medmentions/test.json

DATASET_NAME=medmentions

CONFIG_PATH=./config/biaffinener.conf
CONFIG_NAME=biaffinenermodel_scibertuncased_medmentions

python run_ner.py \
    --gpu 0 \
    --method biaffinener \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --dataset_name ${DATASET_NAME} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype train_and_evaluate

##############
# Method: LLM-NER
# Dataset: CDR
##############

TRAIN=${STORAGE_DATA}/ner/cdr/train.json
DEV=${STORAGE_DATA}/ner/cdr/dev.json
TEST=${STORAGE_DATA}/ner/cdr/test.json

TRAIN_DEMOS=${STORAGE_DATA}/ner/cdr-demos/train.demonstrations.5.count.json
DEV_DEMOS=${STORAGE_DATA}/ner/cdr-demos/dev.demonstrations.5.count.json
TEST_DEMOS=${STORAGE_DATA}/ner/cdr-demos/test.demonstrations.5.count.json

DATASET_NAME=cdr

CONFIG_PATH=./config/llmner.conf
# CONFIG_NAME=llm_llama3_8b_cdr_prompt12fewshot
CONFIG_NAME=openai_gpt4omini_cdr_prompt12fewshot

python run_ner.py \
    --gpu 0 \
    --method llmner \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --dataset_name ${DATASET_NAME} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate

##############
# Method: LLM-NER
# Dataset: Linked-DocRED
##############

TRAIN=${STORAGE_DATA}/ner/linked-docred/train.json
DEV=${STORAGE_DATA}/ner/linked-docred/dev.json
TEST=${STORAGE_DATA}/ner/linked-docred/test.json

TRAIN_DEMOS=${STORAGE_DATA}/ner/linked-docred-demos/train.demonstrations.5.count.json
DEV_DEMOS=${STORAGE_DATA}/ner/linked-docred-demos/dev.demonstrations.5.count.json
TEST_DEMOS=${STORAGE_DATA}/ner/linked-docred-demos/test.demonstrations.5.count.json

DATASET_NAME=linked_docred

CONFIG_PATH=./config/llmner.conf
# CONFIG_NAME=llm_llama3_8b_linked_docred_prompt12fewshot
CONFIG_NAME=openai_gpt4omini_linked_docred_prompt12fewshot

python run_ner.py \
    --gpu 0 \
    --method llmner \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --dataset_name ${DATASET_NAME} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate

##############
# Method: LLM-NER
# Dataset: MedMentions
##############

TRAIN=${STORAGE_DATA}/ner/medmentions/train.json
DEV=${STORAGE_DATA}/ner/medmentions/dev.json
TEST=${STORAGE_DATA}/ner/medmentions/test.json

TRAIN_DEMOS=${STORAGE_DATA}/ner/medmentions-demos/train.demonstrations.5.count.json
DEV_DEMOS=${STORAGE_DATA}/ner/medmentions-demos/dev.demonstrations.5.count.json
TEST_DEMOS=${STORAGE_DATA}/ner/medmentions-demos/test.demonstrations.5.count.json

DATASET_NAME=medmentions

CONFIG_PATH=./config/llmner.conf
# CONFIG_NAME=llm_llama3_8b_medmentions_prompt12fewshot
CONFIG_NAME=openai_gpt4omini_medmentions_prompt12fewshot

python run_ner.py \
    --gpu 0 \
    --method llmner \
    --train ${TRAIN} \
    --dev ${DEV} \
    --test ${TEST} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --dataset_name ${DATASET_NAME} \
    --results_dir ${STORAGE_RESULTS} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --actiontype evaluate

