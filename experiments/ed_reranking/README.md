# ed_reranking

This directory contains example experiments for Entity Disambiguation (reranking phase).

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
experiments/ed_reranking/data/examples/documents.ner.ed_ret.json
experiments/ed_reranking/data/examples/documents_with_supervision.json
experiments/ed_reranking/data/examples/candidate_entities.json
experiments/ed_reranking/data/examples/entity_dict.json
```

### Benchmark datasets

You can also prepare benchmark datasets (e.g., CDR, Linked-DocRED) with the scripts in `experiments/datasets/ed`.

## Step 3. Running the experiments

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in each execution script and adjust them to your environment.

To apply an off-the-shelf ED-reranking component to documents, run:

```bash
bash ./run_ed_reranking.sh
```

To train or evaluate an ED-Reranking component, run:

```bash
bash ./run_ed_reranking_train_eval.sh
```

By default, the script uses the LLM-based ED-Reranking (`METHOD=llm_ed`).

To use BLINK Cross-Encoder (`METHOD=blink_corss_encoder`), change `METHOD` in the script to `blink_cross_encoder`.

If you use the OpenAI API, set your API key in advance.
