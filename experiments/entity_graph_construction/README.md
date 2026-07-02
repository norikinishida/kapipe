# entity-graph-construction

This directory contains example experiments for Entity Garph Construction.

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/entity-graph-construction
pyenv local <your-favorite-env-name>

# 3. Install the KAPipe library
python -m pip install -U kapipe
# or
python -m pip install -e ../..
```

## Step 2. Dataset Preparation

### Example dataset

This directory already includes example data.

```bash
TBA.
```

### Benchmark datasets

You can also prepare benchmark datasets with the scripts in `TBA`. 

## Step 3. Running the Entity Graph Construction component

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

To TBA, run:

```bash
bash ./run_entity_graph_construction.sh
```

By default, the script uses:

TBA.