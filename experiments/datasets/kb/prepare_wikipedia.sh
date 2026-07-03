#!/usr/bin/env bash

# Note that the wikipedia articles are already processed and extracted to JSONL format using the script `experiments/datasets/articles/prepare_wikipedia_articles.sh` before running this script.
WIKIPEDIA=/home/nishida/storage/projects/kapipe/experiments/datasets/articles/wikipedia/enwiki-20240901-pages-articles-multistream.xml.bz2.extracted.processed.jsonl

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/kb


# Results:
#   - STORAGE_DATA/wikipedia/wikipedia20240901.entity_dict.json
python prepare_wikipedia.py \
    --input_file ${WIKIPEDIA} \
    --output_file ${STORAGE_DATA}/wikipedia/wikipedia20240901.entity_dict.json

