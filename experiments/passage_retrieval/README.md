# passage_retrieval

This directory contains example experiments for Passage Retrieval.

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/passage_retrieval
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
experiments/passage_retrieval/data/examples/passages.jsonl
experiments/passage_retrieval/data/examples/questions.json
```

## Step 3. Configuration Setup

Adjust the settings in the following configuration files (HOCON format).

```bash
experiments/passage_retrieval/config/contriever.conf
experiments/passage_retrieval/config/qwen3_embedding.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Then run:

```bash
bash ./run_passage_retrieval.sh
```
