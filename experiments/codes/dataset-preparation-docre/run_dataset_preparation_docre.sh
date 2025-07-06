#/usr/bin/env sh

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph
# CTD=/home/nishida/storage/dataset/CTD
DOCRED=/home/nishida/storage/dataset/DocRED/DocRED
REDOCRED=/home/nishida/storage/dataset/Re-DocRED/Re-DocRED
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED
HOIP=/home/nishida/projects/hoip-dataset/releases/v1

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data


#####################
# CDR
#####################


# Results:
#   - STORAGE_DATA/docre/cdr/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr.py \
        --input_file ${EOG}/data/CDR/processed/${split}_filter.data \
        --output_file ${STORAGE_DATA}/docre/cdr/${split}.json
done

# Results:
#   - STORAGE_DATA/docre/cdr-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/docre/cdr/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task docre \
        --demonstration_pool ${STORAGE_DATA}/docre/cdr/train.json \
        --output_file ${STORAGE_DATA}/docre/cdr-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


#####################
# DocRED
#####################


# Results:
#   - STORAGE_DATA/docre/docred/{train,dev,test}.json
python prepare_docred.py \
    --input_file ${DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/docre/docred/train.json
for split in dev test
do
    python prepare_docred.py \
        --input_file ${DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/docre/docred/${split}.json
done

# Results:
#   - STORAGE_DATA/docre/docred/meta/rel2id.json
#   - STORAGE_DATA/docre/docred/meta/rel_info.json
#   - STORAGE_DATA/docre/docred/meta/ner2id.json
#   - STORAGE_DATA/docre/docred/original/train_annotated.json
#   - STORAGE_DATA/docre/docred/original/train_distant.json
#   - STORAGE_DATA/docre/docred/original/dev.json
mkdir -p ${STORAGE_DATA}/docre/docred/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/docre/docred/meta/
cp ${DOCRED}/rel_info.json ${STORAGE_DATA}/docre/docred/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/docre/docred/meta/
mkdir -p ${STORAGE_DATA}/docre/docred/original
cp ${DOCRED}/train_annotated.json ${STORAGE_DATA}/docre/docred/original/
cp ${DOCRED}/train_distant.json ${STORAGE_DATA}/docre/docred/original/
cp ${DOCRED}/dev.json ${STORAGE_DATA}/docre/docred/original/


#####################
# GDA
#####################


# Results:
#   - STORAGE_DATA/docre/gda/{train,dev,test}.json
for split in train dev test
do
    python prepare_gda.py \
        --input_file ${EOG}/data/GDA/processed/${split}.data \
        --output_file ${STORAGE_DATA}/docre/gda/${split}.json
done


#####################
# HOIP
#####################


# Results:
#   - STORAGE_DATA/docre/hoip-v1/{train,dev,test}.json
for split in train dev test
do
    python prepare_hoip.py \
        --input_file ${HOIP}/${split}.json \
        --output_file ${STORAGE_DATA}/docre/hoip-v1/${split}.json
done

# # Results:
# #   - STORAGE_DATA/docre/hoip-v1/{train,dev,test}.filtered.json
# for split in train dev test
# do
#     python filter_hoip.py \
#         --input_file ${STORAGE_DATA}/docre/hoip-v1/${split}.json \
#         --output_file ${STORAGE_DATA}/docre/hoip-v1/${split}.filtered.json \
#         --target_relations "has result"
# done

# Results:
#   - STORAGE_DATA/docre/hoip-v1-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/docre/hoip-v1/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task docre \
        --demonstration_pool ${STORAGE_DATA}/docre/hoip-v1/train.json \
        --output_file ${STORAGE_DATA}/docre/hoip-v1-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


#####################
# Linked-DocRED
#####################


# Results:
#   - STORAGE_DATA/docre/linked-docred/{train,dev,test}.json
python prepare_linked_docred.py \
    --input_file ${LINKED_DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/docre/linked-docred/train.json
for split in dev test
do
    python prepare_linked_docred.py \
        --input_file ${LINKED_DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/docre/linked-docred/${split}.json
done

# Results:
#   - STORAGE_DATA/docre/linked-docred/meta/rel2id.json
#   - STORAGE_DATA/docre/linked-docred/meta/rel_info.json
#   - STORAGE_DATA/docre/linked-docred/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/docre/linked-docred/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/docre/linked-docred/meta/
cp ${LINKED_DOCRED}/rel_info.json ${STORAGE_DATA}/docre/linked-docred/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/docre/linked-docred/meta/

# Results:
#   - STORAGE_DATA/docre/linked-docred-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/docre/linked-docred/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task docre \
        --demonstration_pool ${STORAGE_DATA}/docre/linked-docred/train.json \
        --output_file ${STORAGE_DATA}/docre/linked-docred-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


#####################
# MedMentions-DSREL (w/ distantly supervised relations)
#####################


# Results:
#   - ${STORAGE_DATA}/docre/medmentions-dsrel/{train,dev,test}.json
python prepare_medmentions_dsrel.py \
    --input_dir ${STORAGE_DATA}/ed/medmentions \
    --triples ${STORAGE_DATA}/kb/umls/umls2017aa.triples_mapped.json \
    --output_dir ${STORAGE_DATA}/docre/medmentions-dsrel

# Results:
#   - STORAGE_DATA/docre/medmentions-dsrel-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/docre/medmentions-dsrel/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task docre \
        --demonstration_pool ${STORAGE_DATA}/docre/medmentions-dsrel/train.json \
        --output_file ${STORAGE_DATA}/docre/medmentions-dsrel-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


#####################
# Re-DocRED
#####################


# Results:
#   - STORAGE_DATA/docre/redocred/{train,dev,test}.json
for split in train dev test
do
    python prepare_docred.py \
        --input_file ${REDOCRED}/data/${split}_revised.json \
        --output_file ${STORAGE_DATA}/docre/redocred/${split}.json
done

# Results:
#   - STORAGE_DATA/docre/redocred/meta/rel2id.json
#   - STORAGE_DATA/docre/redocred/meta/rel_info.json
#   - STORAGE_DATA/docre/redocred/meta/ner2id.json
#   - STORAGE_DATA/docre/redocred/original/train_revised.json
#   - STORAGE_DATA/docre/redocred/original/train_distant.json
#   - STORAGE_DATA/docre/redocred/original/dev_revised.json
#   - STORAGE_DATA/docre/redocred/original/test_revised.json
mkdir -p ${STORAGE_DATA}/docre/redocred/meta
cp ${DOCRED}/DocRED_baseline_metadata/rel2id.json ${STORAGE_DATA}/docre/redocred/meta/
cp ${DOCRED}/rel_info.json ${STORAGE_DATA}/docre/redocred/meta/
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/docre/redocred/meta/
mkdir -p ${STORAGE_DATA}/docre/redocred/original
cp ${REDOCRED}/data/train_revised.json ${STORAGE_DATA}/docre/redocred/original/
cp ${DOCRED}/train_distant.json ${STORAGE_DATA}/docre/redocred/original/
cp ${REDOCRED}/data/dev_revised.json ${STORAGE_DATA}/docre/redocred/original/
cp ${REDOCRED}/data/test_revised.json ${STORAGE_DATA}/docre/redocred/original/


