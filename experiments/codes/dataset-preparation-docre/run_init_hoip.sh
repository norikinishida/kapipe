#/usr/bin/env sh

GO=/home/nishida/storage/dataset/GO/GO.csv
HOIP_ONTOLOGY=/home/nishida/storage/dataset/HOIP-Ontology/HOIP.csv

HOIP_DATASET=/home/nishida/storage/dataset/HOIP-Dataset
HOIP_FILE=COVID_ARDS_36006_lv2_231027


# Results:
#   - HOIP-Dataset/processed-v5/hoip_ontology.json
python init_hoip_ontology.py \
    --input_files ${GO} ${HOIP_ONTOLOGY} \
    --output_file ${HOIP_DATASET}/processed-v5/hoip_ontology.json

# Results:
#   - HOIP-Dataset/processed-v5/HOIP_FILE.triples.aggregated.{train,dev,split}.json
#   - HOIP-Dataset/processed-v5/HOIP_FILE.triples.aggregated.merged.{train,dev,split}.json
#   - HOIP-Dataset/processed-v5/HOIP_FILE.triples.aggregated.merged.hypernyms_marked.{train,dev,split}.json
python init_hoip_dataset.py \
    --input_file ${HOIP_DATASET}/${HOIP_FILE}.csv \
    --process_inconsistency \
    --ontology ${HOIP_DATASET}/processed-v5/hoip_ontology.json \
    --output_dir ${HOIP_DATASET}/processed-v5
