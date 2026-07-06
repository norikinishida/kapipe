# Datasets

This repository describes the procedures for preprocessing benchmark datasets used in experiments.

First, create a Python environment and install the KAPipe library and dependencies.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/datasets
pyenv local <your-favorite-env-name>

# 3. Install the KAPipe library
python -m pip install -U kapipe
# or
python -m pip install -e ../..

# 4. Install dependencies specific to this directory
python -m pip install -r requirements.txt
```

Next, download the original datasets you want to use in your experiments from their official websites. The original datasets must be downloaded before running the scripts below. Check the scripts to see which datasets are expected to be downloaded.

Then, run the scripts corresponding to the datasets you want to use in your experiments. Set `STORAGE_DATA` and the path to the original datasets appropriately for your environment. The scripts will preprocess the original datasets and save them in a format suitable for experiments with this codebase.

## Articles (Corpus)

```bash
cd experiments/datasets/articles

# Wikipedia articles
bash ./prepare_wikipedia_articles.sh

# Wikipedia articles (popular preprocessed corpus)
bash ./prepare_wikipedia_psgs_w100_tsv.sh

# PubMed abstracts
bash ./prepare_pubmed_abstracts.sh

# CDR abstracts
bash ./prepare_cdr_abstracts.sh

# Linked-DocRED articles
bash ./prepare_linked_docred_articles.sh
```

## Knowledge Base

```bash
cd experiments/datasets/kb

# DBPeida (used with Linked-DocRED, etc.)
bash ./prepare_dbpedia.sh

# Wikipedia
# bash ./prepare_wikipedia.sh

# MeSH (used with CDR, etc.)
bash ./prepare_mesh.sh

# UMLS (used with MedMentions, etc.)
bash ./prepare_umls.sh

# HOIP (used with HOIP dataset, etc.)
bash ./prepare_hoip.sh
```

## NER

```bash
cd experiments/datasets/ner

# Linked-DocRED (mentions-only)
bash ./prepare_linked_docred.sh

# CDR (mentions-only)
bash ./prepare_cdr.sh

# MedMentions (mentions-only)
bash ./prepare_medmentions.sh
```

## Entity Disambiguation (candidate retrieval and reranking)

```bash
cd experiments/datasets/ed

# Linked-DocRED (no relations)
bash ./prepare_linked_docred.sh

# CDR (no relations)
bash ./prepare_cdr.sh

# MedMentions
bash ./prepare_medmentions.sh
```

## Document-level Relation Extraction

```bash
cd experiments/datasets/docre

# DocRED / Re-DocRED / Linked-DocRED
bash ./prepare_docred.sh
bash ./prepare_redocred.sh
bash ./prepare_linked_docred.sh

# CDR / GDA
bash ./prepare_cdr.sh
bash ./prepare_gda.sh

# HOIP dataset
bash ./prepare_hoip_dataset.sh
```

## Question Answering

```bash
cd experiments/datasets/qa

# NQ, TriviaQA
bash ./prepare_fb_nq_triviaqa.sh

# PopQA
bash ./prepare_popqa.sh

# BioASQ
bash ./prepare_bioasq.sh
```
