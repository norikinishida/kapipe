#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
# (BLINK Cross-Encoder)
# METHOD=blink_cross_encoder
# CONFIG_PATH=./config/blink_cross_encoder.conf
# CONFIG_NAME=blink_cross_encoder_model_scibertuncased_cdr
# (LLM-ED)
METHOD=llm_ed
CONFIG_PATH=./config/llm_ed.conf
CONFIG_NAME=openai_gpt4omini_cdr_prompt10fewshot

# Input Data
# (In practice, use separate files for training, validation, and test data.
# This example checks whether the model can achieve near 100% accuracy (i.e., overfit) on the training data.)
TRAIN_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
DEV_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
TEST_DOCS=${STORAGE_DATA}/examples/documents_with_triples.json
#
# CANDS_ROOT=${STORAGE_RESULTS}/ed_retrieval/blink_bi_encoder/blink_bi_encoder_model_scibertuncased_cdr/example
# TRAIN_CANDS=${CANDS_ROOT}/train.pred_candidate_entities.json
# DEV_CANDS=${CANDS_ROOT}/dev.pred_candidate_entities.json
# TEST_CANDS=${CANDS_ROOT}/test.pred_candidate_entities.json
# ENTITY_DICT=${CANDS_ROOT}/entity_dict.json
CANDS_ROOT=${STORAGE_DATA}/examples
TRAIN_CANDS=${CANDS_ROOT}/ed_retrieval_results.pred_candidate_entities.json
DEV_CANDS=${CANDS_ROOT}/ed_retrieval_results.pred_candidate_entities.json
TEST_CANDS=${CANDS_ROOT}/ed_retrieval_results.pred_candidate_entities.json
ENTITY_DICT=${CANDS_ROOT}/entity_dict.json
#
TRAIN_DEMOS=${STORAGE_DATA}/examples/documents_without_triples.demonstrations.3.random.json
DEV_DEMOS=${STORAGE_DATA}/examples/documents_without_triples.demonstrations.3.random.json
TEST_DEMOS=${STORAGE_DATA}/examples/documents_without_triples.demonstrations.3.random.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# (BLINK Cross-Encoder)
# python run_ed_reranking_train_eval.py \
#     --method ${METHOD} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --train_documents ${TRAIN_DOCS} \
#     --dev_documents ${DEV_DOCS} \
#     --test_documents ${TEST_DOCS} \
#     --train_candidate_entities ${TRAIN_CANDS} \
#     --dev_candidate_entities ${DEV_CANDS} \
#     --test_candidate_entities ${TEST_CANDS} \
#     --entity_dict ${ENTITY_DICT} \
#     --results_dir ${RESULTS_DIR} \
#     --prefix ${MYPREFIX} \
#     --actiontype train_and_evaluate

# (LLM-ED)
python run_ed_reranking_train_eval.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --train_documents ${TRAIN_DOCS} \
    --dev_documents ${DEV_DOCS} \
    --test_documents ${TEST_DOCS} \
    --train_candidate_entities ${TRAIN_CANDS} \
    --dev_candidate_entities ${DEV_CANDS} \
    --test_candidate_entities ${TEST_CANDS} \
    --train_demonstrations ${TRAIN_DEMOS} \
    --dev_demonstrations ${DEV_DEMOS} \
    --test_demonstrations ${TEST_DEMOS} \
    --entity_dict ${ENTITY_DICT} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX} \
    --actiontype evaluate

