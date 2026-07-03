#/usr/bin/env sh

# Note that the CDR directory is already generated using the script `experiments/datasets/docre/prepare_cdr.sh` before running this script.
CDR=/home/nishida/storage/projects/kapipe/experiments/datasets/docre/cdr

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/articles


# Results:
#   - STORAGE_DATA/articles/cdr/cdr_abstracts.jsonl
python prepare_cdr_abstracts.py \
    --input_dir ${CDR} \
    --output_file ${STORAGE_DATA}/cdr/cdr_abstracts.jsonl

