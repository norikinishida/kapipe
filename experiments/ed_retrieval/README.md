# ed_retrieval

This directory contains example experiments for Entity Disambiguation (Retrieval) (`kapipe.ed_retrieval`).

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/ed_retrieval
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
experiments/ed_retrieval/data/examples/documents_with_typed_mentions.json
experiments/ed_retrieval/data/examples/documents_with_disambiguated_entities.json
experiments/ed_retrieval/data/examples/entity_dict.json
```

### Benchmark datasets

You can also prepare benchmark datasets (e.g., CDR, Linked-DocRED) with the scripts in `experiments/datasets/ed`.

## Step 3. Configuration Setup

Adjust the settings in the following configuration files (HOCON format).

```bash
experiments/ed_retrieval/config/mention_name_entity_retriever.conf
experiments/ed_retrieval/config/blink_bi_encoder.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Then run:

```bash
bash ./run_ed_retrieval.sh
```
