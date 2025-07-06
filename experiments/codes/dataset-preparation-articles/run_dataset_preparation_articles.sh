#/usr/bin/env sh

WIKIPEDIA=/home/nishida/storage/dataset/Wikipedia

STORAGE=/home/nishida/storage/projects/kapipe/experiments
STORAGE_DATA=${STORAGE}/data


#####################
# Wikipedia articles
#####################


# WIKIPEDIA_DUMP_URL=https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles-multistream.xml.bz2
WIKIPEDIA_DUMP_URL=https://dumps.wikimedia.org/enwiki/20240901/enwiki-20240901-pages-articles-multistream.xml.bz2
WIKIPEDIA_DUMP_FILENAME=`basename ${WIKIPEDIA_DUMP_URL}`

# Results:
#   - WIKIPEDIA/WIKIPEDIA_DUMP_FILENAME
mkdir -p ${WIKIPEDIA}
wget ${WIKIPEDIA_DUMP_URL} -P ${WIKIPEDIA}

# Results:
#   - WIKIPEDIA/WIKIPEDIA_DUMP_FILENAME.extracted/*/*
python -m wikiextractor.WikiExtractor ${WIKIPEDIA}/${WIKIPEDIA_DUMP_FILENAME} \
    -o ${STORAGE_DATA}/articles/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted \
    --json \
    --processes 4 \
    -b 1G

# Results:
#   - STORAGE_DATA/articles/wikipedia/WIKIPEDIA_DUMP_FILENAME.extracted.processed.jsonl
python prepare_wikipedia_articles.py \
    --input_dir ${STORAGE_DATA}/articles/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted \
    --output_file ${STORAGE_DATA}/articles/wikipedia/${WIKIPEDIA_DUMP_FILENAME}.extracted.processed.jsonl

#####

# Results:
#   - WIKIPEDIA/psgs_w100.tsv
mkdir -p ${WIKIPEDIA}/fb
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz -P ${WIKIPEDIA}/fb
gzip -d ${WIKIPEDIA}/fb/psgs_w100.tsv.gz

# Results:
#   - STORAGE_DATA/articles/wikipedia/psgs_w100.tsv.processed.jsonl
python prepare_wikipedia_psgs_w100_tsv.py \
    --input_file ${WIKIPEDIA}/fb/psgs_w100.tsv \
    --output_file ${STORAGE_DATA}/articles/wikipedia/psgs_w100.tsv.processed.jsonl


#####################
# PubMed abstracts
#####################


# Results:
#   - STORAGE_DATA/articles/pubmed/pubmed_abstracts.jsonl
python prepare_pubmed_abstracts.py \
    --output_file ${STORAGE_DATA}/articles/pubmed/pubmed_abstracts.jsonl


#####################
# CDR abstracts
#####################


# Results:
#   - STORAGE_DATA/articles/cdr/cdr_abstracts.jsonl
python prepare_cdr_abstracts.py \
    --input_dir ${STORAGE_DATA}/docre/cdr \
    --output_file ${STORAGE_DATA}/articles/cdr/cdr_abstracts.jsonl


#####################
# Linked-DocRED articles
#####################


# Results:
#   - STORAGE_DATA/articles/linked-docred/linked_docred_articles.jsonl
python prepare_linked_docred_articles.py \
    --input_dir ${STORAGE_DATA}/docre/linked-docred \
    --output_file ${STORAGE_DATA}/articles/linked-docred/linked_docred_articles.jsonl



