#!/usr/bin/env sh

# STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE=/home/nishida/projects/kapipe/experiments

STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

# Method
METHOD=llm

# Input Data
INPUT_GRAPH=${STORAGE_DATA}/examples/graph.graphml
INPUT_COMMUNITIES=${STORAGE_DATA}/examples/communities.json

# Output Path
RESULTS_DIR=${STORAGE_RESULTS}
MYPREFIX=example

python run_report_generation.py \
    --method ${METHOD} \
    --input_graph ${INPUT_GRAPH} \
    --input_communities ${INPUT_COMMUNITIES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MYPREFIX}

