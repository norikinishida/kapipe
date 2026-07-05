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
# METHOD=llm_based_report_generator
METHOD=template_based_report_generator

if [ ${METHOD} = "llm_based_report_generator" ]; then
    CONFIG_PATH=./config/llm_based_report_generator.conf
    CONFIG_NAME=llm_cdr
elif [ ${METHOD} = "template_based_report_generator" ]; then
    CONFIG_PATH=./config/template_based_report_generator.conf
    CONFIG_NAME=template_cdr
else
    echo "Invalid method: ${METHOD}"
    exit 1
fi

# Input Data
INPUT_GRAPH=${STORAGE_DATA}/examples/graph.graphml
INPUT_COMMUNITIES=${STORAGE_DATA}/examples/communities.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_report_generation.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_graph ${INPUT_GRAPH} \
    --input_communities ${INPUT_COMMUNITIES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

