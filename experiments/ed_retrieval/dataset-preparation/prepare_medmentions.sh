#/usr/bin/env sh

EOG=/home/nishida/storage/projects/others/fenchri.edge-oriented-graph

DOCRED=/home/nishida/storage/dataset/DocRED/DocRED
LINKED_DOCRED=/home/nishida/storage/dataset/Linked-DocRED/Linked-DocRED/Linked-DocRED

MEDMENTIONS=/home/nishida/storage/dataset/MedMentions/MedMentions

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data


#####################
# CDR
#####################


# Results:
#   - STORAGE_DATA/ed/cdr/{train,dev,test}.json
for split in train dev test
do
    python prepare_cdr.py \
        --input_file ${EOG}/data/CDR/processed/${split}_filter.data \
        --output_file ${STORAGE_DATA}/ed/cdr/${split}.json
    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/ed/cdr/${split}.json \
        --entity_dict ${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json
done

# Results:
#   - STORAGE_DATA/ed/cdr-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/ed/cdr/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task ed \
        --demonstration_pool ${STORAGE_DATA}/ed/cdr/train.json \
        --entity_dict ${STORAGE_DATA}/kb/mesh/mesh2015.entity_dict.json \
        --output_file ${STORAGE_DATA}/ed/cdr-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done


#####################
# Linked-DocRED
#####################


# Results:
#   - STORAGE_DATA/ed/linked-docred/{train,dev,test}.json
python prepare_linked_docred.py \
    --input_file ${LINKED_DOCRED}/train_annotated.json \
    --output_file ${STORAGE_DATA}/ed/linked-docred/train.json
for split in dev test
do
    python prepare_linked_docred.py \
        --input_file ${LINKED_DOCRED}/${split}.json \
        --output_file ${STORAGE_DATA}/ed/linked-docred/${split}.json
done
for split in train dev test
do
    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/ed/linked-docred/${split}.json \
        --entity_dict ${STORAGE_DATA}/kb/dbpedia/dbpedia20200201.entity_dict.json
done

# Results:
#   - STORAGE_DATA/ed/linked-docred/meta/ner2id.json
mkdir -p ${STORAGE_DATA}/ed/linked-docred/meta
cp ${DOCRED}/DocRED_baseline_metadata/ner2id.json ${STORAGE_DATA}/ed/linked-docred/meta/

# Results:
#   - STORAGE_DATA/ed/linked-docred-demos/{train,dev,test}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
N_DEMOS=5
RETRIEVER_METHOD=count
for split in train dev test
do
    python ../dataset-preparation-ner/generate_demonstrations.py \
        --documents ${STORAGE_DATA}/ed/linked-docred/${split}.json \
        --n_demos ${N_DEMOS} \
        --method ${RETRIEVER_METHOD} \
        --task ed \
        --demonstration_pool ${STORAGE_DATA}/ed/linked-docred/train.json \
        --entity_dict ${STORAGE_DATA}/kb/dbpedia/dbpedia20200201.entity_dict.json \
        --output_file ${STORAGE_DATA}/ed/linked-docred-demos/${split}.demonstrations.${N_DEMOS}.${RETRIEVER_METHOD}.json
done

#####################
# MedMentions
#####################


# Results:
#   - STORAGE_DATA/ed/medmentions/{train,dev,test}.json
#   - STORAGE_DATA/ed/medmentions/meta/st21pv_semantic_types.json
python prepare_medmentions.py \
    --input_dir ${MEDMENTIONS} \
    --output_dir ${STORAGE_DATA}/ed/medmentions
for split in train dev test
do
    python test_entity_appearance_in_entity_dict.py \
        --input_file ${STORAGE_DATA}/ed/medmentions/${split}.json \
        --entity_dict ${STORAGE_DATA}/kb/umls/umls2017aa.entity_dict.json
done

