#/usr/bin/env sh

# Note that the Linked-DocRED directory is already generated using the script `experiments/datasets/docre/prepare_linked_docred.sh` before running this script.
LINKED_DOCRED=/home/nishida/storage/projects/kapipe/experiments/datasets/docre/linked_docred

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets


# Results:
#   - STORAGE_DATA/articles/linked_docred/linked_docred_articles.jsonl
python prepare_linked_docred_articles.py \
    --input_dir ${LINKED_DOCRED} \
    --output_file ${STORAGE_DATA}/articles/linked_docred/linked_docred_articles.jsonl



