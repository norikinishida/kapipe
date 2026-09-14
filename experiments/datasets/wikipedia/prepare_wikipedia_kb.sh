#!/usr/bin/env bash

# Note that the Wikipedia articles are already processed and extracted to JSONL format using the script `experiments/datasets/wikipedia/prepare_wikipedia_corpus.sh` before running this script.
WIKIPEDIA=/home/nishida/storage/projects/kapipe/experiments/datasets/wikipedia/corpus/enwiki-20240901-pages-articles-multistream.xml.bz2.extracted.processed.jsonl

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/wikipedia/kb/wikipedia20240901.entity_dict.json
python prepare_wikipedia_kb.py \
    --input_file ${WIKIPEDIA} \
    --output_file ${STORAGE_DATA}/wikipedia/kb/wikipedia20240901.entity_dict.json
