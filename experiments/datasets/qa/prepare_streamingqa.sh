#!/usr/bin/env bash

set -euo pipefail


STREAMINGQA=/home/nishida/storage/dataset/StreamingQA
STREAMINGQA_COMMIT=18dc327429333750fbeca32fd4e27f328127774a

WMT_NEWS_CRAWL=/home/nishida/storage/dataset/WMT-News-Crawl

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download the official extraction code only when it is not already available
# Results:
#   - STREAMINGQA/google-deepmind.streamingqa/
if [ ! -d "${STREAMINGQA}/google-deepmind.streamingqa/.git" ]; then
    mkdir -p "${STREAMINGQA}"
    git clone \
        https://github.com/google-deepmind/streamingqa \
        "${STREAMINGQA}/google-deepmind.streamingqa"
    git -C "${STREAMINGQA}/google-deepmind.streamingqa" \
        switch --detach "${STREAMINGQA_COMMIT}"
fi

# Validate that the current StreamingQA repository is at the expected commit
CURRENT_STREAMINGQA_COMMIT=$(
    git -C "${STREAMINGQA}/google-deepmind.streamingqa" rev-parse HEAD
)
if [ "${CURRENT_STREAMINGQA_COMMIT}" != "${STREAMINGQA_COMMIT}" ]; then
    echo "Unexpected StreamingQA revision: ${CURRENT_STREAMINGQA_COMMIT}" >&2
    echo "Expected revision: ${STREAMINGQA_COMMIT}" >&2
    exit 1
fi

# Download the official QA splits and deduplicated WMT document identifiers
# Results:
#   - STREAMINGQA/wmt_sorting_key_ids.txt.gz
#   - STREAMINGQA/streaminqa_{train,valid,eval}.jsonl
wget -c \
    https://storage.googleapis.com/dm-streamingqa/wmt_sorting_key_ids.txt.gz \
    -P "${STREAMINGQA}"
wget -c \
    https://storage.googleapis.com/dm-streamingqa/streaminqa_train.jsonl.gz \
    -P "${STREAMINGQA}"
wget -c \
    https://storage.googleapis.com/dm-streamingqa/streaminqa_valid.jsonl.gz \
    -P "${STREAMINGQA}"
wget -c \
    https://storage.googleapis.com/dm-streamingqa/streaminqa_eval.jsonl.gz \
    -P "${STREAMINGQA}"

# Download the document-split English WMT News Crawl archives
# Results:
#   - WMT_NEWS_CRAWL/news-docs.{2007..2021}.en.filtered.gz
mkdir -p "${WMT_NEWS_CRAWL}"
for YEAR in $(seq 2007 2021); do
    wget -c \
        "https://data.statmt.org/news-crawl/doc/en/news-docs.${YEAR}.en.filtered.gz" \
        -P "${WMT_NEWS_CRAWL}"
done

# Results:
#   - STREAMINGQA/docs.jsonl
#   - STREAMINGQA/qas.{train,dev,test}.jsonl
#   - STORAGE_DATA/articles/streamingqa/articles.jsonl
#   - STORAGE_DATA/qa/streamingqa/{train,dev,test}.json
#   - STORAGE_DATA/qa/streamingqa/{train,dev,test}.gold_contexts.json
python prepare_streamingqa.py \
    --input_dir "${STREAMINGQA}" \
    --wmt_dir "${WMT_NEWS_CRAWL}" \
    --extraction_file "${STREAMINGQA}/google-deepmind.streamingqa/extraction.py" \
    --output_articles_file "${STORAGE_DATA}/articles/streamingqa/articles.jsonl" \
    --output_questions_dir "${STORAGE_DATA}/qa/streamingqa"

# Results:
#   - STORAGE_DATA/qa/streamingqa/{train,dev,test}_filtered.json
python filter_streamingqa.py \
    --input_questions_dir "${STORAGE_DATA}/qa/streamingqa" \
    --output_questions_dir "${STORAGE_DATA}/qa/streamingqa"

# Results:
#   - STORAGE_DATA/articles/streamingqa/articles_for_test_filtered.jsonl
python reduce_articles_for_streamingqa.py \
    --input_questions_file "${STORAGE_DATA}/qa/streamingqa/test_filtered.json" \
    --input_gold_contexts_file "${STORAGE_DATA}/qa/streamingqa/test.gold_contexts.json" \
    --input_articles_file "${STORAGE_DATA}/articles/streamingqa/articles.jsonl" \
    --index_dir "${STORAGE_DATA}/articles/streamingqa/contriever_index" \
    --output_articles_file "${STORAGE_DATA}/articles/streamingqa/articles_for_test_filtered.jsonl"
