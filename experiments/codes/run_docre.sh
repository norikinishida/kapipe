#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
# (ATLOP)
METHOD=atlop
CONFIG_PATH=./config/atlop.conf
CONFIG_NAME=atlopmodel_scibertcased_cdr_overlap
# (LLM-DocRE)
# METHOD=llmdocre
# CONFIG_PATH=./config/llmdocre.conf
# CONFIG_NAME=openai_gpt4omini_cdr_prompt07fewshot

# Input Data
DATASET_NAME=cdr
TRAIN_DOCS=${STORAGE_DATA}/docre/cdr/train.json
DEV_DOCS=${STORAGE_DATA}/docre/cdr/dev.json
TEST_DOCS=${STORAGE_DATA}/docre/cdr/test.json
#
TRAIN_DEMOS=${STORAGE_DATA}/docre/cdr-demos/train.demonstrations.5.count.json
DEV_DEMOS=${STORAGE_DATA}/docre/cdr-demos/dev.demonstrations.5.count.json
TEST_DEMOS=${STORAGE_DATA}/docre/cdr-demos/test.demonstrations.5.count.json
#
ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (ATLOP)
python run_docre.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --dataset_name ${DATASET_NAME} \
    --train_documents ${TRAIN_DOCS} \
    --dev_documents ${DEV_DOCS} \
    --test_documents ${TEST_DOCS} \
    --results_dir ${RESULTS_DIR} \
    --actiontype train_and_evaluate

# (LLM-DocRE)
# python run_docre.py \
#     --method llmdocre \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --dataset_name ${DATASET_NAME} \
#     --train_documents ${TRAIN_DOCS} \
#     --dev_documents ${DEV_DOCS} \
#     --test_documents ${TEST_DOCS} \
#     --train_demonstrations ${TRAIN_DEMOS} \
#     --dev_demonstrations ${DEV_DEMOS} \
#     --test_demonstrations ${TEST_DEMOS} \
#     --entity_dict ${ENTITY_DICT} \
#     --results_dir ${STORAGE_RESULTS} \
#     --actiontype evaluate
