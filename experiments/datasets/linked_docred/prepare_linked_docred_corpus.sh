#!/usr/bin/env sh

# Note that the Linked-DocRED directory is already generated using the script `experiments/datasets/linked_docred/prepare_linked_docred_docre.sh` before running this script.
LINKED_DOCRED=/home/nishida/storage/projects/kapipe/experiments/datasets/linked_docred/docre

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/linked_docred/corpus/linked_docred_articles.jsonl
python prepare_linked_docred_corpus.py \
    --input_dir ${LINKED_DOCRED} \
    --output_file ${STORAGE_DATA}/linked_docred/corpus/linked_docred_articles.jsonl



