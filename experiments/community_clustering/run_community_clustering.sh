#!/usr/bin/env sh

######
# Storage paths
######

STORAGE_DATA=/home/nishida/projects/kapipe/experiments/community_clustering/data
STORAGE_RESULTS=/home/nishida/projects/kapipe/experiments/community_clustering/results

######
# Experiment configuration
######

# Method
# METHOD=hierarchical_leiden
METHOD=neighborhood_aggregation
# METHOD=triple_level_factorization

if [ "${METHOD}" = "hierarchical_leiden" ]; then
    CONFIG_PATH=./config/hierarchical_leiden.conf
    CONFIG_NAME=size10_lcc
elif [ "${METHOD}" = "neighborhood_aggregation" ]; then
    CONFIG_PATH=./config/neighborhood_aggregation.conf
    CONFIG_NAME=hop1
elif [ "${METHOD}" = "triple_level_factorization" ]; then
    CONFIG_PATH=./config/triple_level_factorization.conf
    CONFIG_NAME=none
else
    echo "Unknown method: ${METHOD}"
    exit 1
fi

# Input Data
INPUT_GRAPH=${STORAGE_DATA}/examples/graph.graphml

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_community_clustering.py \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    --input_graph ${INPUT_GRAPH} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
