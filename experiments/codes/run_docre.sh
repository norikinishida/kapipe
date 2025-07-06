#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
# STORAGE_DATA=${STORAGE}/data
# STORAGE_RESULTS=${STORAGE}/results

# #####################
# # Method: ATLOP
# # Dataset: CDR
# #####################

# TRAIN=${STORAGE_DATA}/docre/cdr/train.json
# DEV=${STORAGE_DATA}/docre/cdr/dev.json
# TEST=${STORAGE_DATA}/docre/cdr/test.json

# DATASET_NAME=cdr

# CONFIG_PATH=./config/atlop.conf
# # CONFIG_NAME=atlopmodel_scibertcased_cdr
# CONFIG_NAME=atlopmodel_scibertcased_cdr_overlap

# python run_docre.py \
#     --gpu 0 \
#     --method atlop \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: ATLOP
# # Dataset: DocRED
# #####################

# TRAIN=${STORAGE_DATA}/docre/docred/train.json
# DEV=${STORAGE_DATA}/docre/docred/dev.json
# TEST=${STORAGE_DATA}/docre/docred/test.json

# DATASET_NAME=docred

# CONFIG_PATH=./config/atlop.conf
# CONFIG_NAME=atlopmodel_bertbasecased_docred

# python run_docre.py \
#     --gpu 0 \
#     --method atlop \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: ATLOP
# # Dataset: GDA
# #####################

# TRAIN=${STORAGE_DATA}/docre/gda/train.json
# DEV=${STORAGE_DATA}/docre/gda/dev.json
# TEST=${STORAGE_DATA}/docre/gda/test.json

# DATASET_NAME=gda

# CONFIG_PATH=./config/atlop.conf
# CONFIG_NAME=atlopmodel_scibertcased_gda

# python run_docre.py \
#     --gpu 0 \
#     --method atlop \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: ATLOP
# # Dataset: Linked-DocRED
# #####################

# TRAIN=${STORAGE_DATA}/docre/linked-docred/train.json
# DEV=${STORAGE_DATA}/docre/linked-docred/dev.json
# TEST=${STORAGE_DATA}/docre/linked-docred/test.json

# DATASET_NAME=linked_docred

# CONFIG_PATH=./config/atlop.conf
# CONFIG_NAME=atlopmodel_bertbasecased_linked_docred

# python run_docre.py \
#     --gpu 0 \
#     --method atlop \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: ATLOP
# # Dataset: MedMentions (w/ distantly supervised relations)
# #####################

# TRAIN=${STORAGE_DATA}/docre/medmentions-dsrel/train.json
# DEV=${STORAGE_DATA}/docre/medmentions-dsrel/dev.json
# TEST=${STORAGE_DATA}/docre/medmentions-dsrel/test.json

# DATASET_NAME=medmentions_dsrel

# CONFIG_PATH=./config/atlop.conf
# # CONFIG_NAME=atlopmodel_scibertcased_medmentions_dsrel
# CONFIG_NAME=atlopmodel_scibertcased_medmentions_dsrel_overlap

# python run_docre.py \
#     --gpu 0 \
#     --method atlop \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: ATLOP
# # Dataset: Re-DocRED
# #####################

# TRAIN=${STORAGE_DATA}/docre/redocred/train.json
# DEV=${STORAGE_DATA}/docre/redocred/dev.json
# TEST=${STORAGE_DATA}/docre/redocred/test.json

# DATASET_NAME=redocred

# CONFIG_PATH=./config/atlop.conf
# CONFIG_NAME=atlopmodel_bertbasecased_redocred

# python run_docre.py \
#     --gpu 0 \
#     --method atlop \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: Mention-Agnostic ATLOP (MA-ATLOP)
# # Dataset: CDR
# #####################

# TRAIN=${STORAGE_DATA}/docre/cdr/train.json
# DEV=${STORAGE_DATA}/docre/cdr/dev.json
# TEST=${STORAGE_DATA}/docre/cdr/test.json

# ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

# DATASET_NAME=cdr

# CONFIG_PATH=./config/maatlop.conf
# CONFIG_NAME=maatlopmodel_scibertcased_cdr
# # CONFIG_NAME=maatlopmodel_scibertcased_cdr_mc

# python run_docre.py \
#     --gpu 0 \
#     --method maatlop \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --entity_dict ${ENTITY_DICT} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: Mention-Agnostic ATLOP (MA-ATLOP)
# # Dataset: HOIP
# #####################

# TRAIN=${STORAGE_DATA}/docre/hoip-v1/train.json
# DEV=${STORAGE_DATA}/docre/hoip-v1/dev.json
# TEST=${STORAGE_DATA}/docre/hoip-v1/test.json

# ENTITY_DICT=${STORAGE_DATA}/kb/hoip-kb/hoip.entity_dict.json

# DATASET_NAME=hoip

# CONFIG_PATH=./config/maatlop.conf
# CONFIG_NAME=maatlopmodel_scibertcased_hoip
# # CONFIG_NAME=maatlopmodel_scibertcased_hoip_mc

# python run_docre.py \
#     --gpu 0 \
#     --method maatlop \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --entity_dict ${ENTITY_DICT} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: Mention-Agnostic QA BERT (MA-QA)
# # Dataset: CDR
# #####################

# TRAIN=${STORAGE_DATA}/docre/cdr/train.json
# DEV=${STORAGE_DATA}/docre/cdr/dev.json
# TEST=${STORAGE_DATA}/docre/cdr/test.json

# ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

# DATASET_NAME=cdr

# CONFIG_PATH=./config/maqa.conf
# CONFIG_NAME=maqamodel_scibertcased_cdr
# # CONFIG_NAME=maqamodel_scibertcased_cdr_mc

# python run_docre.py \
#     --gpu 0 \
#     --method maqa \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --entity_dict ${ENTITY_DICT} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: Mention-Agnostic QA BERT (MA-QA)
# # Dataset: HOIP
# #####################

# TRAIN=${STORAGE_DATA}/docre/hoip-v1/train.json
# DEV=${STORAGE_DATA}/docre/hoip-v1/dev.json
# TEST=${STORAGE_DATA}/docre/hoip-v1/test.json

# ENTITY_DICT=${STORAGE_DATA}/kb/hoip-kb/hoip.entity_dict.json

# DATASET_NAME=hoip

# CONFIG_PATH=./config/maqa.conf
# CONFIG_NAME=maqamodel_scibertcased_hoip
# # CONFIG_NAME=maqamodel_scibertcased_hoip_mc

# python run_docre.py \
#     --gpu 0 \
#     --method maqa \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --entity_dict ${ENTITY_DICT} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype train_and_evaluate

# #####################
# # Method: LLM-DocRE
# # Dataset: CDR
# #####################

# TRAIN=${STORAGE_DATA}/docre/cdr/train.json
# DEV=${STORAGE_DATA}/docre/cdr/dev.json
# TEST=${STORAGE_DATA}/docre/cdr/test.json

# TRAIN_DEMOS=${STORAGE_DATA}/docre/cdr-demos/train.demonstrations.5.count.json
# DEV_DEMOS=${STORAGE_DATA}/docre/cdr-demos/dev.demonstrations.5.count.json
# TEST_DEMOS=${STORAGE_DATA}/docre/cdr-demos/test.demonstrations.5.count.json

# ENTITY_DICT=${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

# DATASET_NAME=cdr

# CONFIG_PATH=./config/llmdocre.conf
# # CONFIG_NAME=llm_llama3_8b_cdr_prompt07fewshot
# CONFIG_NAME=openai_gpt4omini_cdr_prompt07fewshot

# python run_docre.py \
#     --gpu 0 \
#     --method llmdocre \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --train_demonstrations ${TRAIN_DEMOS} \
#     --dev_demonstrations ${DEV_DEMOS} \
#     --test_demonstrations ${TEST_DEMOS} \
#     --entity_dict ${ENTITY_DICT} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype evaluate

# #####################
# # Method: LLM-DocRE
# # Dataset: HOIP
# #####################

# TRAIN=${STORAGE_DATA}/docre/hoip-v1/train.json
# DEV=${STORAGE_DATA}/docre/hoip-v1/dev.json
# TEST=${STORAGE_DATA}/docre/hoip-v1/test.json

# TRAIN_DEMOS=${STORAGE_DATA}/docre/hoip-v1-demos/train.demonstrations.5.random.json
# DEV_DEMOS=${STORAGE_DATA}/docre/hoip-v1-demos/dev.demonstrations.5.random.json
# TEST_DEMOS=${STORAGE_DATA}/docre/hoip-v1-demos/test.demonstrations.5.random.json

# ENTITY_DICT=${STORAGE_DATA}/kb/hoip-kb/hoip.entity_dict.json

# DATASET_NAME=hoip

# CONFIG_PATH=./config/llmdocre.conf
# CONFIG_NAME=llm_llama3_8b_hoip_prompt07fewshot_cn

# python run_docre.py \
#     --gpu 0 \
#     --method llmdocre \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --train_demonstrations ${TRAIN_DEMOS} \
#     --dev_demonstrations ${DEV_DEMOS} \
#     --test_demonstrations ${TEST_DEMOS} \
#     --entity_dict ${ENTITY_DICT} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype evaluate

# #####################
# # Method: LLM-DocRE
# # Dataset: Linked-DocRED
# #####################

# TRAIN=${STORAGE_DATA}/docre/linked-docred/train.json
# DEV=${STORAGE_DATA}/docre/linked-docred/dev.json
# TEST=${STORAGE_DATA}/docre/linked-docred/test.json

# TRAIN_DEMOS=${STORAGE_DATA}/docre/linked-docred-demos/train.demonstrations.5.count.json
# DEV_DEMOS=${STORAGE_DATA}/docre/linked-docred-demos/dev.demonstrations.5.count.json
# TEST_DEMOS=${STORAGE_DATA}/docre/linked-docred-demos/test.demonstrations.5.count.json

# ENTITY_DICT=${STORAGE_DATA}/kb/dbpedia/dbpedia20200201.entity_dict.json

# DATASET_NAME=linked_docred

# CONFIG_PATH=./config/llmdocre.conf
# CONFIG_NAME=openai_gpt4omini_linked_docred_prompt07fewshot

# python run_docre.py \
#     --gpu 0 \
#     --method llmdocre \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --train_demonstrations ${TRAIN_DEMOS} \
#     --dev_demonstrations ${DEV_DEMOS} \
#     --test_demonstrations ${TEST_DEMOS} \
#     --entity_dict ${ENTITY_DICT} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype evaluate

# #####################
# # Method: LLM-DocRE
# # Dataset: MedMentions (w/ distantly supervised relations)
# #####################

# TRAIN=${STORAGE_DATA}/docre/medmentions-dsrel/train.json
# DEV=${STORAGE_DATA}/docre/medmentions-dsrel/dev.json
# TEST=${STORAGE_DATA}/docre/medmentions-dsrel/test.json

# ENTITY_DICT=${STORAGE_DATA}/kb/umls/umls2017aa.entity_dict.json

# TRAIN_DEMOS=${STORAGE_DATA}/docre/medmentions-dsrel-demos/train.demonstrations.5.count.json
# DEV_DEMOS=${STORAGE_DATA}/docre/medmentions-dsrel-demos/dev.demonstrations.5.count.json
# TEST_DEMOS=${STORAGE_DATA}/docre/medmentions-dsrel-demos/test.demonstrations.5.count.json

# DATASET_NAME=medmentions_dsrel

# CONFIG_PATH=./config/llmdocre.conf
# # CONFIG_NAME=llm_llama3_8b_medmentions_dsrel_prompt07fewshot
# CONFIG_NAME=openai_gpt4omini_medmentions_dsrel_prompt07fewshot

# python run_docre.py \
#     --gpu 0 \
#     --method llmdocre \
#     --train ${TRAIN} \
#     --dev ${DEV} \
#     --test ${TEST} \
#     --train_demonstrations ${TRAIN_DEMOS} \
#     --dev_demonstrations ${DEV_DEMOS} \
#     --test_demonstrations ${TEST_DEMOS} \
#     --entity_dict ${ENTITY_DICT} \
#     --dataset_name ${DATASET_NAME} \
#     --results_dir ${STORAGE_RESULTS} \
#     --config_path ${CONFIG_PATH} \
#     --config_name ${CONFIG_NAME} \
#     --actiontype evaluate
