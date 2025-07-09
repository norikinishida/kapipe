#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
# (Biaffine-NER)
# METHOD=biaffinener
# CONFIG_PATH=./config/biaffinener.conf
# CONFIG_NAME=biaffinenermodel_scibertuncased_cdr
# (LLM-NER)
METHOD=llmner
CONFIG_PATH=./config/llmner.conf
CONFIG_NAME=openai_gpt4omini_cdr_prompt13fewshot

# Input Data
# (In practice, use separate files for training, validation, and test data.
# This example checks whether the model can achieve near 100% accuracy (i.e., overfit) on the training data.)
DATASET_NAME=cdr
TRAIN_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
DEV_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
TEST_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
#
TRAIN_DEMOS=${STORAGE_DATA}/examples/documents_without_triples.demonstrations.3.random.json
DEV_DEMOS=${STORAGE_DATA}/examples/documents_without_triples.demonstrations.3.random.json
TEST_DEMOS=${STORAGE_DATA}/examples/documents_without_triples.demonstrations.3.random.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (Biaffine-NER)
# python run_ner.py \
#     --method ${METHOD} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --dataset_name ${DATASET_NAME} \
#     --train_documents ${TRAIN_DOCS} \
#     --dev_documents ${DEV_DOCS} \
#     --test_documents ${TEST_DOCS} \
#     --results_dir ${STORAGE_RESULTS} \
#     --prefix ${MYPREFIX} \
#     --actiontype train_and_evaluate

# (LLM-NER)
python run_ner.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --dataset_name ${DATASET_NAME} \
    --train_documents ${TRAIN_DOCS} \
    --dev_documents ${DEV_DOCS} \
    --test_documents ${TEST_DOCS} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --results_dir ${STORAGE_RESULTS} \
    --prefix ${MYPREFIX} \
    --actiontype evaluate

