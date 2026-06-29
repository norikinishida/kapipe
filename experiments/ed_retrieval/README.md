# Entity Disambiguation (candidate retrieval)

This directory contains example experiments for Entity Disambiguation (candidate retrieval phase).

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
experiments/ed_retrieval/data/examples/entity_dict.json
experiments/ed_retrieval/data/examples/documents.json
experiments/ed_retrieval/data/examples/documents.ner.json
experiments/ed_retrieval/data/examples/documents_with_supervision.json
```

### Benchmark datasets

If necessary, you can prepare benchmark datasets with the following commands.

```bash
cd experiments/ed_retrieval/dataset-preparation

bash ./prepare_wikidata.sh
bash ./prepare_mesh.sh

bash ./prepare_linked_docred.sh
bash ./prepare_cdr.sh
bash ./prepare_medmentions.sh
```

These commands extract each dataset under `experiments/ed_retrieval/data/`.

Example:
```bash
experiments/ed_retrieval/data/mesh/mesh2015.entity_dict.json
experiments/ed_retrieval/data/cdr/train.json
experiments/ed_retrieval/data/cdr/dev.json
experiments/ed_retrieval/data/cdr/test.json
```

## Step 3. Running the ED-Retrieval Component

First, check `STORAGE` in each execution script and adjust it to your environment.

To apply an off-the-shelf ED-Retrieval component to documents, run:

```bash
bash ./run_ed_retrieval.sh
```

To train or evaluate an ED-Retrieval component, run:

```bash
bash ./run_ed_retrieval_train_eval.sh
```

By default, both scripts use the BLINK Bi-Encoder method (`method_name = blink_bi_encoder`).
