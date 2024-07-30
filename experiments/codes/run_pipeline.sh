#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

IDENTIFIER=cdr_biaffinener_blink_atlop

DEV=${STORAGE_DATA}/docre/cdr/dev.json
TEST=${STORAGE_DATA}/docre/cdr/test.json

python run_pipeline.py \
    --identifier ${IDENTIFIER} \
    --dev ${DEV} \
    --test ${TEST} \
    --results_dir ${STORAGE_RESULTS}

