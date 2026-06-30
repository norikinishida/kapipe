#!/usr/bin/env bash

WIKIPEDIA=/home/nishida/storage/dataset/Wikipedia

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/articles


# WIKIPEDIA_DUMP_URL=https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles-multistream.xml.bz2
WIKIPEDIA_DUMP_URL=https://dumps.wikimedia.org/enwiki/20240901/enwiki-20240901-pages-articles-multistream.xml.bz2
WIKIPEDIA_DUMP_FILENAME=`basename ${WIKIPEDIA_DUMP_URL}`


# Results:
#   - WIKIPEDIA/WIKIPEDIA_DUMP_FILENAME
mkdir -p ${WIKIPEDIA}
wget ${WIKIPEDIA_DUMP_URL} -P ${WIKIPEDIA}


# Results:
#   - WIKIPEDIA/WIKIPEDIA_DUMP_FILENAME.extracted/*/*
python -m wikiextractor.WikiExtractor ${WIKIPEDIA}/${WIKIPEDIA_DUMP_FILENAME} \
    -o ${STORAGE_DATA}/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted \
    --json \
    --processes 4 \
    -b 1G


# Results:
#   - STORAGE_DATA/wikipedia/WIKIPEDIA_DUMP_FILENAME.extracted.processed.jsonl
python prepare_wikipedia_articles.py \
    --input_dir ${STORAGE_DATA}/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted \
    --output_file ${STORAGE_DATA}/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted.processed.jsonl

