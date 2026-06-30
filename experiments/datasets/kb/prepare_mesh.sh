#!/usr/bin/env bash

MESH=/home/nishida/storage/dataset/MeSH/2015/xmlmesh

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/kb


# Results:
#   - STORAGE_DATA/mesh/mesh2015.entity_dict.json
python prepare_mesh.py \
    --input_dir ${MESH} \
    --output_file ${STORAGE_DATA}/mesh/mesh2015.entity_dict.json

