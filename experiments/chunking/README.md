# Chunking

This directory contains example experiments for text chunking.

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
experiments/chunking/data/examples/articles.jsonl
```

Each input passage should be a JSON object with a `text` field.

```json
{"title": "Example title", "text": "Example passage text."}
```

The `title` field is optional.

## Step 3. Running the Chunking Component

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

To split passages into chunks, run:

```bash
bash ./run_chunking.sh
```

By default, the script uses:

```bash
CONFIG_NAME=en_core_sci_md_w100
```
