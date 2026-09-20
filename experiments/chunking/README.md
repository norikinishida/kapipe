# Chunking

This directory contains example experiments for text chunking (`kapipe.chunking`).

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/chunking
pyenv local <your-favorite-env-name>

# 3. Install the KAPipe library
python -m pip install -U kapipe
# or
python -m pip install -e ../..

# 4. Install dependencies specific to this directory
python -m pip install -r requirements.txt
```

If you use a spaCy model, install it in the same environment.

```bash
python -m spacy download en_core_web_md
```

If you use SciSpaCy, install the model required by your configuration.

## Step 2. Dataset Preparation

### Example dataset

This directory already includes example data.

```bash
experiments/chunking/data/examples/corpus/articles.jsonl
```

Each input passage should be a JSON object with `passage_key` and `text` fields.

```json
{"passage_key": "passage#001", "title": "Example title", "text": "Example passage text."}
```

The `title` field is optional.

## Step 3. Configuration Setup

Adjust the settings in the following configuration file (HOCON format).

```bash
experiments/chunking/config/default.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Then run:

```bash
bash ./run_chunking.sh
```
