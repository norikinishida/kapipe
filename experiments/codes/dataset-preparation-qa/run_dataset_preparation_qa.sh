#/usr/bin/env sh

NQ=/home/nishida/storage/dataset/NQ
TRIVIAQA=/home/nishida/storage/dataset/TriviaQA
BIOASQ_TRAINDEV=/home/nishida/storage/dataset/BioASQ/BioASQ12/BioASQ-training12b/training12b_new.json

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data

SIZE=256


#####################
# NQ
#####################


# # Results:
# #   - STORAGE_DATA/qa/nq/dev.json
# #   - STORAGE_DATA/qa/nq/dev_${SIZE}.json
# python prepare_nq.py \
#     --input_file ${NQ}/v1.0-simplified_simplified-nq-dev-all.jsonl.gz \
#     --output_file ${STORAGE_DATA}/qa/nq/dev.json

# python prepare_nq.py \
#     --input_file ${NQ}/v1.0-simplified_simplified-nq-dev-all.jsonl.gz \
#     --output_file ${STORAGE_DATA}/qa/nq/dev_${SIZE}.json \
#     --size ${SIZE}

# # Results:
# #   - STORAGE_DATA/qa/nq/dev_${SIZE}.contexts_as_passages.json
# python extract_contexts_as_passages_from_qa_dataset.py \
#     --input_file ${STORAGE_DATA}/qa/nq/dev_${SIZE}.json \
#     --output_file ${STORAGE_DATA}/qa/nq/dev_${SIZE}.contexts_as_passages.jsonl

# # Results:
# #   - STORAGE_DATA/qa/nq/dev_${SIZE}.contexts_as_passages.chunked.json
# python ../dataset-preparation-articles/chunk_passages.py \
#     --input_file ${STORAGE_DATA}/qa/nq/dev_${SIZE}.contexts_as_passages.jsonl \
#     --output_file ${STORAGE_DATA}/qa/nq/dev_${SIZE}.contexts_as_passages.chunked.jsonl \
#     --window_size 200

######

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
#   - STORAGE_DATA/qa/nq/{train1,dev1,test1}.json
#   - STORAGE_DATA/qa/nq/{train1,dev1,test1}_${SIZE}.json
for split in train dev test
do
    python prepare_fb_nq_triviaqa.py \
        --input_file ${NQ}/fb/nq-${split}.qa.csv \
        --output_file ${STORAGE_DATA}/qa/nq/${split}1.json
    python prepare_fb_nq_triviaqa.py \
        --input_file ${NQ}/fb/nq-${split}.qa.csv \
        --output_file ${STORAGE_DATA}/qa/nq/${split}1_${SIZE}.json \
        --size ${SIZE}
done

# Results
#   - STORAGE_DATA/qa/nq/{train2,dev2}.json
#   - STORAGE_DATA/qa/nq/{train2,dev2}.{gold,distant}_contexts.json
#   - STORAGE_DATA/qa/nq/{train2,dev2}_${SIZE}.json
#   - STORAGE_DATA/qa/nq/{train2,dev2}_${SIZE}.{gold,distant}_contexts.json
for split in train dev
do
    for context_type in gold distant
    do
        python prepare_fb_nq_triviaqa.py \
            --input_file ${NQ}/fb/biencoder-nq-${split}.json \
            --from_json \
            --context_type ${context_type} \
            --output_file ${STORAGE_DATA}/qa/nq/${split}2.json
        python prepare_fb_nq_triviaqa.py \
            --input_file ${NQ}/fb/biencoder-nq-${split}.json \
            --from_json \
            --context_type ${context_type} \
            --output_file ${STORAGE_DATA}/qa/nq/${split}2_${SIZE}.json \
            --size ${SIZE}
    done
done


#####################
# TriviaQA
#####################


# # Results
# #   - STORAGE_DATA/qa/triviaqa/{train,dev,test}.json
# #   - STORAGE_DATA/qa/triviaqa/dev_${SIZE}.json
# python prepare_triviaqa.py \
#     --output_dir ${STORAGE_DATA}/qa/triviaqa \
#     --size ${SIZE}

# # Results
# #   - STORAGE_DATA/qa/triviaqa/dev_${SIZE}.contexts_as_passages.jsonl
# python extract_contexts_as_passages_from_qa_dataset.py \
#     --input_file ${STORAGE_DATA}/qa/triviaqa/dev_${SIZE}.json \
#     --output_file ${STORAGE_DATA}/qa/triviaqa/dev_${SIZE}.contexts_as_passages.jsonl \
#     --with_title

# # Results
# #   - STORAGE_DATA/qa/triviaqa/dev_${SIZE}.contexts_as_passages.chunked.jsonl
# python ../dataset-preparation-articles/chunk_passages.py \
#     --input_file ${STORAGE_DATA}/qa/triviaqa/dev_${SIZE}.contexts_as_passages.jsonl \
#     --output_file ${STORAGE_DATA}/qa/triviaqa/dev_${SIZE}.contexts_as_passages.chunked.jsonl \
#     --window_size 200

######

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
#   - STORAGE_DATA/qa/triviaqa/{dev1,test1}.json
#   - STORAGE_DATA/qa/triviaqa/{dev1,test1}_${SIZE}.json
for split in dev test
do
    python prepare_fb_nq_triviaqa.py \
        --input_file ${TRIVIAQA}/fb/trivia-${split}.qa.csv \
        --output_file ${STORAGE_DATA}/qa/triviaqa/${split}1.json
    python prepare_fb_nq_triviaqa.py \
        --input_file ${TRIVIAQA}/fb/trivia-${split}.qa.csv \
        --output_file ${STORAGE_DATA}/qa/triviaqa/${split}1_${SIZE}.json \
        --size ${SIZE}
done

# Results
#   - STORAGE_DATA/qa/triviaqa/dev2.json
#   - STORAGE_DATA/qa/triviaqa/dev2.{distant}_contexts.json
#   - STORAGE_DATA/qa/triviaqa/dev2_${SIZE}.json
#   - STORAGE_DATA/qa/triviaqa/dev2_${SIZE}.{distant}_contexts.json
for context_type in distant
do
    python prepare_fb_nq_triviaqa.py \
        --input_file ${TRIVIAQA}/fb/biencoder-trivia-dev.json \
        --from_json \
        --context_type ${context_type} \
        --output_file ${STORAGE_DATA}/qa/triviaqa/dev2.json
    python prepare_fb_nq_triviaqa.py \
        --input_file ${TRIVIAQA}/fb/biencoder-trivia-dev.json \
        --from_json \
        --context_type ${context_type} \
        --output_file ${STORAGE_DATA}/qa/triviaqa/dev2_${SIZE}.json \
        --size ${SIZE}
done


#####################
# BioASQ
#####################


# Results
#   - STORAGE_DATA/qa/bioasq/train_dev.json
#   - STORAGE_DATA/qa/bioasq/train_dev.contexts.json
python prepare_bioasq.py \
    --input_file ${BIOASQ_TRAINDEV} \
    --output_file ${STORAGE_DATA}/qa/bioasq/train_dev.json

# Results
#   - STORAGE_DATA/qa/bioasq/train_dev_list_only_pubmed_limited_${SIZE}.json
#   - STORAGE_DATA/qa/bioasq/train_dev_list_only_pubmed_limited_${SIZE}.contexts.json
python prepare_bioasq.py \
    --input_file ${BIOASQ_TRAINDEV} \
    --output_file ${STORAGE_DATA}/qa/bioasq/train_dev_list_only_pubmed_limited_${SIZE}.json \
    --target_answer_types list \
    --pubmed_abstracts ${STORAGE_DATA}/articles/pubmed/pubmed_abstracts.jsonl \
    --size ${SIZE}

# Results
#   - STORAGE_DATA/qa/bioasq/passages.jsonl
python extract_passages.py \
    --input_file ${STORAGE_DATA}/qa/bioasq/train_dev.contexts.json \
    --output_file ${STORAGE_DATA}/qa/bioasq/passages.jsonl


#####################
# PopQA
#####################


# Results
#   - STORAGE_DATA/qa/popqa/test.json
#   - STORAGE_DATA/qa/popqa/test_${SIZE}.json
python prepare_popqa.py \
    --output_dir ${STORAGE_DATA}/qa/popqa \
    --size ${SIZE}

