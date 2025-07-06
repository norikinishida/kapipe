#/usr/bin/env sh


DBPEDIA=/home/nishida/storage/dataset/DBPedia/2020.02.01
MESH=/home/nishida/storage/dataset/MeSH/2015/xmlmesh
UMLS=/home/nishida/storage/dataset/UMLS/2017AA
HOIP=/home/nishida/projects/hoip-dataset/releases/v1

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data


#####################
# Wikidata
#####################


# Results:
#   - DBPedia/long_abstracts_lang_en.ttl
mkdir -p ${DBPEDIA}
wget https://downloads.dbpedia.org/repo/dbpedia/text/long-abstracts/2020.02.01/long-abstracts_lang=en.ttl.bz2 -P ${DBPEDIA}/
mv "${DBPEDIA}/long-abstracts_lang=en.ttl.bz2" ${DBPEDIA}/long-abstracts_lang_en.ttl.bz2
bzip2 -d ${DBPEDIA}/long-abstracts_lang_en.ttl.bz2

# Results:
#   - STORAGE_DATA/kb/dbpedia/dbpedia20200201.entity_dict.json
python prepare_dbpedia_entity_dict.py \
    --input_file ${DBPEDIA}/long-abstracts_lang_en.ttl \
    --output_file ${STORAGE_DATA}/kb/dbpedia/dbpedia20200201.entity_dict.json

# # Results:
# #   - STORAGE_DATA/kb/wikipedia/wikipedia20240901.entity_dict.json
# python prepare_wikipedia_entity_dict.py \
#     --input_file ${STORAGE_DATA}/articles/wikipedia/enwiki-20240901-pages-articles-multistream.xml.bz2.extracted.processed.jsonl \
#     --output_file ${STORAGE_DATA}/kb/wikipedia/wikipedia20240901.entity_dict.json


#####################
# Mesh
#####################


# Results:
#   - STORAGE_DATA/kb/mesh/mesh2015.entity_dict.json
#   - STORAGE_DATA/kb/mesh/mesh2015.entity_embs.*.json
python prepare_mesh.py \
    --input_dir ${MESH} \
    --output_file ${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json


#####################
# UMLS
#####################


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


#####################
# HOIP Ontology
#####################


# Results:
#   - STORAGE_DATA/kb/hoip-kb/hoip.entity_dict.json
python prepare_hoip_kb.py \
    --input_file ${HOIP}/hoip_ontology.json \
    --output_file ${STORAGE_DATA}/kb/hoip-kb/hoip.entity_dict.json

