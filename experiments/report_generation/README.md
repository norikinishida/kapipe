# report_generation

This directory contains example experiments for Report Generation (`kapipe.report_generation`).

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/report_generation
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
experiments/report_generation/data/examples/graph.graphml
experiments/report_generation/data/examples/communities.json
```

## Step 3. Configuration Setup

Adjust the settings in the following configuration files (HOCON format).

```bash
experiments/report_generation/config/llm_based_report_generator.conf
experiments/report_generation/config/template_based_report_generator.conf
```

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Then run:

```bash
bash ./run_report_generation.sh
```

If you use the OpenAI API, set your API key in advance.
