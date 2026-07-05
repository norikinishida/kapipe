# rag_pipeline

This directory contains example experiments for RAG pipeline.

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/rag_pipeline
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
experiments/rag_pipeline/data/examples/passages.jsonl
experiments/rag_pipeline/data/examples/questions.json
experiments/rag_pipeline/data/examples/questions_with_answers.json
experiments/rag_pipeline/data/examples/questions.gold_contexts.json
```

## Step 3. Configuration Setup

Adjust the settings in the following configuration file (HOCON format).

```bash
experiments/rag_pipeline/config/default.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

The pipeline consists of two sequential actions.

- `indexing`: build a retrieval index over passages.
- `inference`: retrieve passages for each question and generate answers

Run one action at a time:

```bash
bash ./run_rag_pipeline.sh --actiontype indexing
```

Run all actions:

```bash
bash ./run_rag_pipeline.sh --actiontype all
```

If you use the OpenAI API, set your API key in advance.
