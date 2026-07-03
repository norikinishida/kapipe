#!/usr/bin/env bash

HOIP=/home/nishida/storage/dataset/HOIP-Dataset

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/docre


# Results:
#  - HOIP/hoip-dataset/releases/v1/{train,dev,test}.json
#  - HOIP/hoip-dataset/releases/v1/hoip_ontology.json
mkdir -p ${HOIP}
cd ${HOIP}
git clone https://github.com/norikinishida/hoip-dataset.git
cd hoip-dataset/releases
tar -zxvf v1.tar.gz


# Results:
#   - STORAGE_DATA/docre/hoip_v1/{train,dev,test}.json
for split in train dev test
do
    python prepare_hoip_dataset.py \
        --input_file ${HOIP}/hoip-dataset/releases/v1/${split}.json \
        --output_file ${STORAGE_DATA}/hoip_v1/${split}.json
done


# # Results:
# #   - STORAGE_DATA/docre/hoip_v1/{train,dev,test}.filtered.json
# for split in train dev test
# do
#     python filter_hoip.py \
#         --input_file ${STORAGE_DATA}/hoip_v1/${split}.json \
#         --output_file ${STORAGE_DATA}/hoip_v1/${split}.filtered.json \
#         --target_relations "has result"
# done

