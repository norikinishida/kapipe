#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
METHOD=hierarchical_leiden
# METHOD=neighborhood_aggregation
# METHOD=triple_level_factorization

# Input Data
INPUT_GRAPH=${STORAGE_DATA}/examples/graph.graphml

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_community_clustering.py \
    --method ${METHOD} \
    --input_graph ${INPUT_GRAPH} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}
