#!/usr/bin/env bash

UMLS=/home/nishida/storage/dataset/UMLS/2017AA

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/kb/umls/umls2017aa.entity_dict.json
#   - STORAGE_DATA/kb/umls/umls2017aa.triples.json
#   - STORAGE_DATA/kb/umls/umls2017aa.entity_type_dict.json
python prepare_umls.py \
    --input_dir ${UMLS} \
    --output_file ${STORAGE_DATA}/kb/umls/umls2017aa.entity_dict.json


# Results:
#   - STORAGE_DATA/kb/umls/umls2017aa.triples_mapped.json
python map_umls_relation_labels.py \
    --input_triples ${STORAGE_DATA}/kb/umls/umls2017aa.triples.json \
    --output_triples ${STORAGE_DATA}/kb/umls/umls2017aa.triples_mapped.json


# Results:
#   - STORAGE_DATA/kb/umls/umls2017aa.triples_mapped_filtered.json
python filter_umls.py \
    --input_triples ${STORAGE_DATA}/kb/umls/umls2017aa.triples_mapped.json \
    --entity_dict ${STORAGE_DATA}/kb/umls/umls2017aa.entity_dict.json \
    --output_triples ${STORAGE_DATA}/kb/umls/umls2017aa.triples_mapped_filtered.json

