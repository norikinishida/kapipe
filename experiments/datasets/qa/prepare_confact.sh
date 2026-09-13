#!/usr/bin/env bash

set -euo pipefail


CONFACT=/home/nishida/storage/dataset/CONFACT
CONFACT_REVISION=05a1227b4821ebf1b843f0ce5824f9420006cfdb
MODC_FILENAME=ModC.pkl.gz
HUMC_FILENAME=HumC.pkl.gz
MODC_SHA256=427d0418f22172601d52aec1fe56048323283959a5efea1a6cf1faf25f6ea893
HUMC_SHA256=1ca8bc4e7a20c210287ccce587bce4d9477e9d2dc84219d244655965e21556dc
CONFACT_BASE_URL="https://raw.githubusercontent.com/zoeyyes/CONFACT/${CONFACT_REVISION}/data/dataset"

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Download both official CONFACT evaluation splits from a fixed revision
# Results:
#   - CONFACT/MODC_FILENAME
#   - CONFACT/HUMC_FILENAME
mkdir -p "${CONFACT}"
wget -c \
    "${CONFACT_BASE_URL}/${MODC_FILENAME}" \
    -O "${CONFACT}/${MODC_FILENAME}"
wget -c \
    "${CONFACT_BASE_URL}/${HUMC_FILENAME}" \
    -O "${CONFACT}/${HUMC_FILENAME}"

# Validate the integrity of the downloaded files using SHA-256 checksums
echo "${MODC_SHA256}  ${CONFACT}/${MODC_FILENAME}" \
    | sha256sum --check --status
echo "${HUMC_SHA256}  ${CONFACT}/${HUMC_FILENAME}" \
    | sha256sum --check --status

# Results:
#   - STORAGE_DATA/articles/confact/articles.jsonl
#   - STORAGE_DATA/qa/confact/{modc,humc}.json
#   - STORAGE_DATA/qa/confact/{modc,humc}.gold_contexts.json
python prepare_confact.py \
    --input_modc_file "${CONFACT}/${MODC_FILENAME}" \
    --input_humc_file "${CONFACT}/${HUMC_FILENAME}" \
    --output_articles_file "${STORAGE_DATA}/articles/confact/articles.jsonl" \
    --output_dir "${STORAGE_DATA}/qa/confact"
