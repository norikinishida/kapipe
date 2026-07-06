#!/usr/bin/env bash

DBPEDIA=/home/nishida/storage/dataset/DBPedia/2020.02.01

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - DBPEDIA/long_abstracts_lang_en.ttl
mkdir -p ${DBPEDIA}
wget https://downloads.dbpedia.org/repo/dbpedia/text/long-abstracts/2020.02.01/long-abstracts_lang=en.ttl.bz2 -P ${DBPEDIA}/
mv "${DBPEDIA}/long-abstracts_lang=en.ttl.bz2" ${DBPEDIA}/long-abstracts_lang_en.ttl.bz2
bzip2 -d ${DBPEDIA}/long-abstracts_lang_en.ttl.bz2


# Results:
#   - STORAGE_DATA/kb/dbpedia/dbpedia20200201.entity_dict.json
python prepare_dbpedia.py \
    --input_file ${DBPEDIA}/long-abstracts_lang_en.ttl \
    --output_file ${STORAGE_DATA}/kb/dbpedia/dbpedia20200201.entity_dict.json

