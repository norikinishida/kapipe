# community_clustering

This directory contains example experiments for Community Clustering.

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/community_clustering
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
experiments/community_clustering/data/examples/graph.graphml
```

The input graph must be a GraphML file readable by `networkx.read_graphml`.

## Step 3. Running the experiments

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

Then run:

```bash
bash ./run_community_clustering.sh
```

By default, the script uses Neighborhood Aggregation method (with hop-size=1).

To use different methods (e.g., Hierarchical Leiden, Triple-Level Factorization), change `METHOD` in the script.