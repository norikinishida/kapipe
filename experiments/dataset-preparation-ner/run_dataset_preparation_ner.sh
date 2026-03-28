#/usr/bin/env sh

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph

CONLL03=/home/nishida/storage/dataset/CoNLL-2003

DOCRED=/home/nishida/storage/dataset/DocRED/DocRED
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

MEDMENTIONS=/home/nishida/storage/dataset/MedMentions/MedMentions

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data


#####################
# CDR
#####################


# Results:
#   - STORAGE_DATA/ner/cdr/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr.py \
        --input_file ${EOG}/data/CDR/processed/${split}_filter.data \
        --output_file ${STORAGE_DATA}/ner/cdr/${split}.json
done

# Results:
#   - STORAGE_DATA/ner/cdr-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python generate_demonstrations.py \
        --documents ${STORAGE_DATA}/ner/cdr/${split}.json \
        --n_demos ${N_DEMOS} \
        --task ner \
        --method ${RETRIEVER_METHOD} \
        --demonstration_pool ${STORAGE_DATA}/ner/cdr/train.json \
        --output_file ${STORAGE_DATA}/ner/cdr-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


#####################
# CoNLL 2003
#####################


# Results:
#   - STORAGE_DATA/ner/conll2003/{train,testa,testb}.json
#   - STORAGE_DATA/ner/conll2003/meta/entity_type_to_id.json
for split in train testa testb
do
    python prepare_conll2003.py \
        --input_file ${CONLL03}/eng.${split} \
        --output_file ${STORAGE_DATA}/ner/conll2003/${split}.json
done

# Results:
#   - STORAGE_DATA/ner/conll2003-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train testa testb
do
    python generate_demonstrations.py \
        --documents ${STORAGE_DATA}/ner/conll2003/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task ner \
        --demonstration_pool ${STORAGE_DATA}/ner/conll2003/train.json \
        --output_file ${STORAGE_DATA}/ner/conll2003-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


#####################
# Linked-DocRED
#####################


# Results:
#   - STORAGE_DATA/ner/linked-docred/{train,dev,test}.json
python prepare_linked_docred.py \
    --input_file ${LINKED_DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/ner/linked-docred/train.json
for split in dev test
do
    python prepare_linked_docred.py \
        --input_file ${LINKED_DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/ner/linked-docred/${split}.json
done

# Results:
#   - STORAGE_DATA/ner/linked-docred/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/ner/linked-docred/meta
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/ner/linked-docred/meta/

# Results:
#   - STORAGE_DATA/ner/linked-docred-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python generate_demonstrations.py \
        --documents ${STORAGE_DATA}/ner/linked-docred/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task ner \
        --demonstration_pool ${STORAGE_DATA}/ner/linked-docred/train.json \
        --output_file ${STORAGE_DATA}/ner/linked-docred-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


#####################
# MedMentions
#####################


# Results:
#   - STORAGE_DATA/ner/medmentions/{train,dev,test}.json
#   - STORAGE_DATA/ner/medmentions/meta/st21pv_semantic_types.json
python prepare_medmentions.py \
    --input_dir ${MEDMENTIONS} \
    --output_dir ${STORAGE_DATA}/ner/medmentions

# Results:
#   - STORAGE_DATA/ner/medmentions-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python generate_demonstrations.py \
        --documents ${STORAGE_DATA}/ner/medmentions/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task ner \
        --demonstration_pool ${STORAGE_DATA}/ner/medmentions/train.json \
        --output_file ${STORAGE_DATA}/ner/medmentions-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done

