#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
METHOD=template

# Input Data
INPUT_GRAPH=${STORAGE_DATA}/examples/graph.graphml
INPUT_COMMUNITIES=${STORAGE_DATA}/examples/communities.json
NODE_ATTR_KEYS="name entity_type description"
EDGE_ATTR_KEYS="relation"

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example4

python run_report_generation.py \
    --method ${METHOD} \
    --input_graph ${INPUT_GRAPH} \
    --input_communities ${INPUT_COMMUNITIES} \
    --node_attr_keys ${NODE_ATTR_KEYS} \
    --edge_attr_keys ${EDGE_ATTR_KEYS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

