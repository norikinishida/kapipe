#/usr/bin/env sh

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph
DOCRED=/home/nishida/storage/dataset/DocRED/DocRED
REDOCRED=/home/nishida/storage/dataset/Re-DocRED/Re-DocRED
CTD=/home/nishida/storage/dataset/CTD
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
RETRIEVER_METHOD=first
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/docre/cdr/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --demonstration_pool ${STORAGE_DATA}/docre/cdr/train.json \
        --output_file ${STORAGE_DATA}/docre/cdr-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


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
#   - STORAGE_DATA/kb/hoip-kb/hoip.entity_dict.json
python prepare_hoip_kb.py \
    --input_file ${HOIP}/hoip_ontology.json \
    --output_file ${STORAGE_DATA}/kb/hoip-kb/hoip.entity_dict.json

# Results:
#   - STORAGE_DATA/docre/hoip-v1-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=first
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/docre/hoip-v1/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --demonstration_pool ${STORAGE_DATA}/docre/hoip-v1/train.json \
        --output_file ${STORAGE_DATA}/docre/hoip-v1-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done

