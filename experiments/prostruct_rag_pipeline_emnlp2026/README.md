# ProStructRAG (EMNLP 2026)

This directory provides the codebase used in the ProStruct-RAG experiments in the following paper:

- Nishida et al., EMNLP 2026, Beyond Retrieval: Structuring Evolving and Inconsistent External Knowledge with Proposition Relations for RAG. (to appear)

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/prostruct_rag_pipeline_emnlp2026
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

This directory already contains example data.

```bash
experiments/prostruct_rag_pipeline_emnlp2026/data/examples/articles.jsonl
experiments/prostruct_rag_pipeline_emnlp2026/data/examples/questions.json
```

`articles.jsonl` contains source passages, and each passage requires `passage_key`, `title` (optional), and `text`.
`questions.json` contains questions, and each question requires `question_key` and `question`; QA evaluation additionally uses `answers`.
When temporal processing is enabled in the configuration, the `timestamp` fields in articles and questions are used.

### Asteria dataset

We provide the Asteria dataset used in the above paper in the following paths.

```bash
experiments/prostruct_rag_pipeline_emnlp2026/data/asteria/propositions.jsonl
experiments/prostruct_rag_pipeline_emnlp2026/data/asteria/questions.json
```

## Step 3. Configuration Setup

Adjust the settings in the following configuration file (HOCON format).

```bash
experiments/prostruct_rag_pipeline_emnlp2026/config/default.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution scripts and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Run each action sequentially using the following commands:

```bash
bash ./run_prostruct_rag_pipeline.sh --actiontype proposition_extraction
bash ./run_prostruct_rag_pipeline.sh --actiontype proposition_relation_extraction
bash ./run_prostruct_rag_pipeline.sh --actiontype proposition_relation_refinement
bash ./run_prostruct_rag_pipeline.sh --actiontype passage_graph_construction
bash ./run_prostruct_rag_pipeline.sh --actiontype passage_retrieval_indexing
bash ./run_prostruct_rag_pipeline.sh --actiontype inference
```

Run all actions at once:

```bash
bash ./run_prostruct_rag_pipeline.sh --actiontype all
```

If you use the OpenAI API, set your API key in advance.
