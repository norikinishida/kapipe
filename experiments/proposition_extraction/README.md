# Proposition Extraction

This directory contains example experiments for proposition extraction (`kapipe.proposition_extraction`).

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/proposition_extraction
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
experiments/proposition_extraction/data/examples/corpus/articles.jsonl
```

## Step 3. Configuration Setup

Adjust the settings in the following configuration file (HOCON format).

```bash
experiments/proposition_extraction/config/llm_proposition_extraction.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Then run:

```bash
bash ./run_proposition_extraction.sh
```

If you use the OpenAI API, set your API key in advance.

To use the OpenAI Batch API, submit the requests and fetch the results after the batches complete.
Batch IDs are saved as `*.batch_ids.json` in the experiment output directory.

```bash
bash ./run_proposition_extraction.sh --batch_mode submit
bash ./run_proposition_extraction.sh --batch_mode fetch
```
