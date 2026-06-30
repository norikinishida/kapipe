#!/usr/bin/env bash

NQ=/home/nishida/storage/dataset/NQ
TRIVIAQA=/home/nishida/storage/dataset/TriviaQA

STORAGE_DATA=/home/nishida/storage/projects/kapipe/experiments/datasets/qa

SIZE=256


# Results
#   - NQ/fb/nq-{train,dev,test}.qa.csv
#   - NQ/fb/biencoder-nq-{train,dev}.json
mkdir -p ${NQ}/fb
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv -P ${NQ}/fb
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv -P ${NQ}/fb
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv -P ${NQ}/fb
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz -P ${NQ}/fb
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz -P ${NQ}/fb
gzip -d ${NQ}/fb/biencoder-nq-train.json.gz
gzip -d ${NQ}/fb/biencoder-nq-dev.json.gz


# Results
#   - STORAGE_DATA/nq/{train1,dev1,test1}.json
#   - STORAGE_DATA/nq/{train1,dev1,test1}_${SIZE}.json
for split in train dev test
do
    python prepare_fb_nq_triviaqa.py \
        --input_file ${NQ}/fb/nq-${split}.qa.csv \
        --output_file ${STORAGE_DATA}/nq/${split}1.json

    python prepare_fb_nq_triviaqa.py \
        --input_file ${NQ}/fb/nq-${split}.qa.csv \
        --output_file ${STORAGE_DATA}/nq/${split}1_${SIZE}.json \
        --size ${SIZE}
done

# Results
#   - STORAGE_DATA/nq/{train2,dev2}.json
#   - STORAGE_DATA/nq/{train2,dev2}.{gold,distant}_contexts.json
#   - STORAGE_DATA/nq/{train2,dev2}_${SIZE}.json
#   - STORAGE_DATA/nq/{train2,dev2}_${SIZE}.{gold,distant}_contexts.json
for split in train dev
do
    for context_type in gold distant
    do
        python prepare_fb_nq_triviaqa.py \
            --input_file ${NQ}/fb/biencoder-nq-${split}.json \
            --from_json \
            --context_type ${context_type} \
            --output_file ${STORAGE_DATA}/nq/${split}2.json

        python prepare_fb_nq_triviaqa.py \
            --input_file ${NQ}/fb/biencoder-nq-${split}.json \
            --from_json \
            --context_type ${context_type} \
            --output_file ${STORAGE_DATA}/nq/${split}2_${SIZE}.json \
            --size ${SIZE}
    done
done


# Results
#   - TRIVIAQA/fb/trivia-{dev,test}.qa.csv
#   - TRIVIAQA/fb/biencoder-trivia-dev.json
mkdir -p ${TRIVIAQA}/fb
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz -P ${TRIVIAQA}/fb
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz -P ${TRIVIAQA}/fb
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz -P ${TRIVIAQA}/fb
gzip -d ${TRIVIAQA}/fb/trivia-dev.qa.csv.gz
gzip -d ${TRIVIAQA}/fb/trivia-test.qa.csv.gz
gzip -d ${TRIVIAQA}/fb/biencoder-trivia-dev.json.gz


# Results
#   - STORAGE_DATA/triviaqa/{dev1,test1}.json
#   - STORAGE_DATA/triviaqa/{dev1,test1}_${SIZE}.json
for split in dev test
do
    python prepare_fb_nq_triviaqa.py \
        --input_file ${TRIVIAQA}/fb/trivia-${split}.qa.csv \
        --output_file ${STORAGE_DATA}/triviaqa/${split}1.json
    python prepare_fb_nq_triviaqa.py \
        --input_file ${TRIVIAQA}/fb/trivia-${split}.qa.csv \
        --output_file ${STORAGE_DATA}/triviaqa/${split}1_${SIZE}.json \
        --size ${SIZE}
done


# Results
#   - STORAGE_DATA/triviaqa/dev2.json
#   - STORAGE_DATA/triviaqa/dev2.{distant}_contexts.json
#   - STORAGE_DATA/triviaqa/dev2_${SIZE}.json
#   - STORAGE_DATA/triviaqa/dev2_${SIZE}.{distant}_contexts.json
for context_type in distant
do
    python prepare_fb_nq_triviaqa.py \
        --input_file ${TRIVIAQA}/fb/biencoder-trivia-dev.json \
        --from_json \
        --context_type ${context_type} \
        --output_file ${STORAGE_DATA}/triviaqa/dev2.json
    python prepare_fb_nq_triviaqa.py \
        --input_file ${TRIVIAQA}/fb/biencoder-trivia-dev.json \
        --from_json \
        --context_type ${context_type} \
        --output_file ${STORAGE_DATA}/triviaqa/dev2_${SIZE}.json \
        --size ${SIZE}
done

