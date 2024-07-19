#/usr/bin/env sh

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph
MESH=/home/nishida/storage/dataset/MeSH/2015/xmlmesh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data


#####################
# CDR & Mesh
#####################


# Results:
#   - STORAGE_DATA/kb/mesh/mesh2015.entity_dict.json
#   - STORAGE_DATA/kb/mesh/mesh2015.entity_embs.*.json
python prepare_mesh.py \
    --input_dir ${MESH} \
    --output_file ${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json

# Results:
#   - STORAGE_DATA/ed/cdr/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr.py \
        --input_file ${EOG}/data/CDR/processed/${split}_filter.data \
        --output_file ${STORAGE_DATA}/ed/cdr/${split}.json
    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/ed/cdr/${split}.json \
        --entity_dict ${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json
done

# Results:
#   - STORAGE_DATA/ed/cdr-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=first
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/ed/cdr/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --demonstration_pool ${STORAGE_DATA}/ed/cdr/train.json \
        --output_file ${STORAGE_DATA}/ed/cdr-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done

