# graphrag_pipeline_tacl2026

This directory contains example experiments for the GraphRAG pipeline.

この実験ディレクトリは、以下の論文で使われたコードベースです:

- [Nishida et al., TACL 2026, **Dissecting GraphRAG: A Modular Analysis of Knowledge Structuring for Factoid Question Answering**.](https://aclanthology.org/2026.tacl-1.29/)

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/graphrag_pipeline_tacl2026
pyenv local <your-favorite-env-name>

# 3. Install the KAPipe library
python -m pip install -U kapipe
# or
python -m pip install -e ../..

# 4. Install dependencies specific to this directory
python -m pip install -r requirements.txt
```

## Step 2. Dataset Preparation

### Example dataset

This directory already includes example data.

```bash
experiments/graphrag_pipeline_tacl2026/data/examples/documents.json
experiments/graphrag_pipeline_tacl2026/data/examples/documents_with_triples.json
experiments/graphrag_pipeline_tacl2026/data/examples/entity_dict.json
experiments/graphrag_pipeline_tacl2026/data/examples/questions.json
experiments/graphrag_pipeline_tacl2026/data/examples/questions_with_answers.json
```

## Step 3. Configuration Setup

Adjust the settings in the following configuration file (HOCON format).

```bash
experiments/graphrag_pipeline_tacl2026/config/default.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

The pipeline consists of seven sequential actions.

- `triple_extraction`: extract triples from documents using NER, ED-Retrieval, ED-Reranking, and DocRE components.
- `entity_graph_construction`: build an entity graph from triples.
- `community_clustering`: cluster entities in the graph.
- `report_generation`: generate reports for the clustered communities.
- `chunking`: split community reports into chunks.
- `retrieval_indexing`: build a passage retrieval index over chunked reports.
- `inference`: retrieve report chunks for each question and generate answers.

Run one action at a time:

```bash
bash ./run_graphrag_pipeline.sh --actiontype triple_extraction
```

Run all actions:

```bash
bash ./run_graphrag_pipeline.sh --actiontype all
```

If you use the OpenAI API, set your API key in advance.
