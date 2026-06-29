#/usr/bin/env sh

MESH=/home/nishida/storage/dataset/MeSH/2015/xmlmesh

STORAGE=/home/nishida/storage/projects/kapipe/experiments/ed_retrieval
STORAGE_DATA=${STORAGE}/data

# Results:
#   - STORAGE_DATA/mesh/mesh2015.entity_dict.json
python prepare_mesh.py \
    --input_dir ${MESH} \
    --output_file ${STORAGE_DATA}/mesh/mesh2015.entity_dict.json

