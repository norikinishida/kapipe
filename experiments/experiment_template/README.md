# EXPERIMENT_NAME

This directory contains example experiments for EXPERIMENT_NAME.

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/EXPERIMENT_NAME
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
experiments/EXPERIMENT_NAME/data/examples/SOMETHING
```

### Benchmark datasets

You can also prepare benchmark datasets with the scripts in `TBA`. 

## Step 3. Running the experiments

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

Then run:

```bash
bash ./run_experiment.sh
```

By default, the script uses METHOD_NAME_A (`METHOD=METHOD_NAME_A`).

To use METHOD_NAME_B (`METHOD=METHOD_NAME_B`), change `METHOD` in the script to `METHOD_NAME_B`.

If you use the OpenAI API, set your API key in advance.
