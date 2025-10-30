#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Input Data
DOCUMENTS=${STORAGE_DATA}/examples/documents_with_triples.json
ADDITIONAL_TRIPLES=${STORAGE_DATA}/examples/additional_triples.json
ENTITY_DICT=${STORAGE_DATA}/examples/entity_dict.json

# Output Data
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

# With Entity Dictionary
python run_knowledge_graph_construction.py \
    --documents_list ${DOCUMENTS} \
    --entity_dict ${ENTITY_DICT} \
    --additional_triples ${ADDITIONAL_TRIPLES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

# # Without Entity Dictionary
# python run_knowledge_graph_construction.py \
#     --documents_list ${DOCUMENTS} \
#     --additional_triples ${ADDITIONAL_TRIPLES} \
#     --results_dir ${RESULTS_DIR} \
#     --prefix ${MYPREFIX}
