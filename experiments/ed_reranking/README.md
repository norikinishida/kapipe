# ed_reranking

This directory contains example experiments for Entity Disambiguation (Reranking) (`kapipe.ed_reranking`).

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/ed_reranking
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
experiments/ed_reranking/data/examples/ed/documents_with_disambiguated_entities.json
experiments/ed_reranking/data/examples/misc/candidate_entities.json
experiments/ed_reranking/data/examples/kb/entity_dict.json
```

### Benchmark datasets

You can also prepare benchmark datasets (e.g., CDR, Linked-DocRED) with the scripts in `experiments/datasets/`.

## Step 3. Configuration Setup

Adjust the settings in the following configuration files (HOCON format).

```bash
experiments/ed_reranking/config/identical_entity_reranker.conf
experiments/ed_reranking/config/blink_cross_encoder.conf
experiments/ed_reranking/config/llm_ed.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Then run:

```bash
bash ./run_ed_reranking.sh
```

If you use the OpenAI API, set your API key in advance.

To use the OpenAI Batch API, submit the requests and fetch the results after the batches complete.
Batch IDs are saved as `*.batch_ids.json` in the experiment output directory.

```bash
bash ./run_ed_reranking.sh --batch_mode submit
bash ./run_ed_reranking.sh --batch_mode fetch
```
