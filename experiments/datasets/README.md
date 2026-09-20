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

# 4. Install dependencies for datasets other than FRAMES
python -m pip install -r requirements.txt
```

Next, download the original datasets you want to use in your experiments from their official websites. The original datasets must be downloaded before running the scripts below. Check the scripts to see which datasets are expected to be downloaded.

Then, run the scripts corresponding to the datasets you want to use in your experiments. Set `STORAGE_DATA` and the path to the original datasets appropriately for your environment. The scripts will preprocess the original datasets and save them in a format suitable for experiments with this codebase.

## Corpus-only

### CDR abstracts:

```bash
cd experiments/datasets/cdr

# Corpus (used with CDR-QA)
bash ./prepare_cdr_corpus.sh
```

### Linked-DocRED articles:

```bash
cd experiments/datasets/linked_docred

# Corpus (used with Linked-DocRED-QA)
bash ./prepare_linked_docred_corpus.sh
```

### PubMed:

```bash
cd experiments/datasets/pubmed

# Corpus
bash ./prepare_pubmed_corpus.sh
```

### Wikipedia:

```bash
cd experiments/datasets/wikipedia

# Corpus
bash ./prepare_wikipedia_corpus.sh

# Corpus for DPR (used with NQ, TriviaQA, HotpotQA, 2WikiMultiHopQA, MuSiQue)
bash ./prepare_wikipedia_corpus_psgs_w100.sh
```

## Information Extraction and Knowledge Base

### CDR:

```bash
cd experiments/datasets/cdr

# NER
bash ./prepare_cdr_ner.sh
# Entity Disambiguation
bash ./prepare_cdr_ed.sh
# Document-level Relation Extraction
bash ./prepare_cdr_docre.sh
```

### CoNLL 2003:

```bash
cd experiments/datasets/conll2003

# NER
bash ./prepare_conll2003_ner.sh
```

### DBPedia:

```bash
cd experiments/datasets/dbpedia

# KB (used with Linked-DocRED)
bash ./prepare_dbpedia_kb.sh
```

### DocRED:

```bash
cd experiments/datasets/docred

# Document-level Relation Extraction
bash ./prepare_docred_docre.sh
```

### GDA:

```bash
cd experiments/datasets/gda

# Document-level Relation Extraction
bash ./prepare_gda_docre.sh
```

### HOIP:

```bash
cd experiments/datasets/hoip

# KB (used with HOIP dataset)
bash ./prepare_hoip_kb.sh

# Document-level Relation Extraction (mention-agnostic)
bash ./prepare_hoip_docre.sh
```

### Linked-DocRED:

```bash
cd experiments/datasets/linked_docred

# NER
bash ./prepare_linked_docred_ner.sh
# Entity Disambiguation
bash ./prepare_linked_docred_ed.sh
# Document-level Relation Extraction
bash ./prepare_linked_docred_docre.sh
```

### MedMentions:

```bash
cd experiments/datasets/medmentions

# NER
bash ./prepare_medmentions_ner.sh
# Entity Disambiguation
bash ./prepare_medmentions_ed.sh
```

### MeSH:

```bash
cd experiments/datasets/mesh

# KB (used with CDR)
bash ./prepare_mesh_kb.sh
```

### Re-DocRED:

```bash
cd experiments/datasets/redocred

# Document-level Relation Extraction
bash ./prepare_redocred_docre.sh
```

### UMLS:

```bash
cd experiments/datasets/umls

# KB (used with MedMentions)
bash ./prepare_umls_kb.sh
```

### Wikipedia:

```bash
cd experiments/datasets/wikipedia

# KB
bash ./prepare_wikipedia_kb.sh
```

## Question Answering, Fact Checking, etc.

### 2WikiMultiHopQA:

```bash
cd experiments/datasets/2wikimultihopqa

# QA
bash ./prepare_2wikimultihopqa_qa.sh
```

### BioASQ:

```bash
cd experiments/datasets/bioasq

# QA and corpus
bash ./prepare_bioasq_qa_and_corpus.sh
```

### BrowseComp-Plus:

```bash
cd experiments/datasets/browsecomp_plus

# QA and corpus
bash ./prepare_browsecomp_plus_qa_and_corpus.sh
```

### CLARK-News:

```bash
cd experiments/datasets/clark_news

# QA and corpus
bash ./prepare_clark_news_qa_and_corpus.sh
```

### CONFACT:

```bash
cd experiments/datasets/confact

# QA and corpus
bash ./prepare_confact_qa_and_corpus.sh
```

### ConfRAG:

```bash
cd experiments/datasets/confrag

# QA and corpus
bash ./prepare_confrag_qa_and_corpus.sh
```

### FanOutQA:

```bash
cd experiments/datasets/fanoutqa

# QA and corpus
bash ./prepare_fanoutqa_qa_and_corpus.sh
```

### FEVER:

```bash
cd experiments/datasets/fever

# QA and corpus
bash ./prepare_fever_qa_and_corpus.sh
```

### FRAMES:

NOTE: Use an independent Python environment for FRAMES because its dependencies conflict with `requirements.txt`.

```bash
cd experiments/datasets/frames

# Set up an independent Python environment for FRAMES
pyenv virtualenv 3.11.14 <your-frames-env-name>
pyenv local <your-frames-env-name>
python -m pip install -e ../../..
python -m pip install -r ./requirements_frames.txt

# QA and corpus
bash ./prepare_frames_qa_and_corpus.sh
```

### HotpotQA:

```bash
cd experiments/datasets/hotpotqa

# QA
bash ./prepare_hotpotqa_qa.sh
```

### LongBench v2:

```bash
cd experiments/datasets/longbench_v2

# QA and corpus
bash ./prepare_longbench_v2_qa_and_corpus.sh
```

### LongMemEval:

```bash
cd experiments/datasets/longmemeval

# QA
bash ./prepare_longmemeval_qa.sh
```

### Loong:

```bash
cd experiments/datasets/loong

# QA and corpus
bash ./prepare_loong_qa_and_corpus.sh
```

### MDCR:

```bash
cd experiments/datasets/mdcr

# QA and corpus
bash ./prepare_mdcr_qa_and_corpus.sh
```

### MuSiQue:

```bash
cd experiments/datasets/musique

# QA
bash ./prepare_musique_qa.sh
```

### Natural Questions:

```bash
cd experiments/datasets/nq

# QA
bash ./prepare_dpr_nq_qa.sh
```

### PopQA:

```bash
cd experiments/datasets/popqa

# QA
bash ./prepare_popqa_qa.sh
```

### StreamingQA:

```bash
cd experiments/datasets/streamingqa

# QA and corpus
bash ./prepare_streamingqa_qa_and_corpus.sh
```

### TriviaQA:

```bash
cd experiments/datasets/triviaqa

# QA
bash ./prepare_dpr_triviaqa_qa.sh
```

