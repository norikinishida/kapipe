# Proposition Relation Refinement

This directory contains example experiments for proposition relation refinement (`kapipe.proposition_relation_refinement`).

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/proposition_relation_refinement
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
experiments/proposition_relation_refinement/data/examples/triples.json
```

Each input triple should be a JSON object with `head`, `relation`, `tail`, and `explanation` fields.

```json
{
    "head": {"text": "Example head proposition.", "timestamp": "2024-06-01"},
    "relation": "supports",
    "tail": {"text": "Example tail proposition.", "timestamp": "2023-05-01"},
    "explanation": "Example explanation."
}
```

The `timestamp` field in each proposition is optional.

## Step 3. Configuration Setup

Adjust the settings in the following configuration file (HOCON format).

```bash
experiments/proposition_relation_refinement/config/llm_proposition_relation_refiner.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Then run:

```bash
bash ./run_proposition_relation_refinement.sh
```

If you use the OpenAI API, set your API key in advance.
