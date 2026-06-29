#!/usr/bin/env bash

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph/data/CDR/processed

STORAGE=/home/nishida/storage/projects/kapipe/experiments/ed_retrieval
STORAGE_DATA=${STORAGE}/data

# Results:
#   - STORAGE_DATA/ed_retrieval/cdr/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr.py \
        --input_file ${EOG}/${split}_filter.data \
        --output_file ${STORAGE_DATA}/cdr/${split}.json
    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/cdr/${split}.json \
        --entity_dict ${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json
done

# Results:
#   - STORAGE_DATA/ed/cdr-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/ed/cdr/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task ed \
        --demonstration_pool ${STORAGE_DATA}/ed/cdr/train.json \
        --entity_dict ${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json \
        --output_file ${STORAGE_DATA}/ed/cdr-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done
