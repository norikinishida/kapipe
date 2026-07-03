#!/usr/bin/env sh

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/report_generation/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/report_generation/results

######
# Experiment configuration
######

# Method
# METHOD=llm
METHOD=template

if [ ${METHOD} = "llm" ]; then
    CONFIG_PATH=./config/llm.conf
    CONFIG_NAME=llm_cdr
elif [ ${METHOD} = "template" ]; then
    CONFIG_PATH=./config/template.conf
    CONFIG_NAME=template_cdr
else
    echo "Invalid method: ${METHOD}"
    exit 1
fi

# Input Data
INPUT_GRAPH=${STORAGE_DATA}/examples/graph.graphml
INPUT_COMMUNITIES=${STORAGE_DATA}/examples/communities.json
NODE_ATTR_KEYS="name entity_type description"
EDGE_ATTR_KEYS="relation"

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_report_generation.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_graph ${INPUT_GRAPH} \
    --input_communities ${INPUT_COMMUNITIES} \
    --node_attr_keys ${NODE_ATTR_KEYS} \
    --edge_attr_keys ${EDGE_ATTR_KEYS} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

