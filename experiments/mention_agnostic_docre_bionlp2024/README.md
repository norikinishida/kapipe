# mention_agnostic_docre_bionlp2024

This directory contains example experiments for mention-agnostic document-level relation extraction (`kapipe.docre.MAATLOP` and `kapipe.docre.MAQA`).

This experiment directory also provides the codebase used in the following paper:

- [Oumaima and Nishida et al., BioNLP 2024, Mention-Agnostic Information Extraction for Ontological Annotation of Biomedical Articles.](https://aclanthology.org/2024.bionlp-1.37/)

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/mention_agnostic_docre_bionlp2024
pyenv local <your-favorite-env-name>

# 3. Install the KAPipe library
python -m pip install -U kapipe
# or
python -m pip install -e ../..

# 4. Install dependencies specific to this directory
python -m pip install -r requirements.txt
```

## Step 2. Dataset Preparation

### CDR dataset

The CDR dataset and the MeSH entity dictionary can be prepared using the scripts in `experiments/datasets/cdr/prepare_cdr_docre.sh` and `experiments/datasets/mesh/prepare_mesh_kb.sh`.

### HOIP dataset

The HOIP dataset and the HOIP entity dictionary can be prepared using the scripts in `experiments/datasets/hoip/prepare_hoip_docre.sh` and `experiments/datasets/hoip/prepare_hoip_kb.sh`.

## Step 3. Configuration Setup

Adjust the settings in the following configuration files (HOCON format).

```bash
experiments/mention_agnostic_docre_bionlp2024/config/ma_atlop.conf
experiments/mention_agnostic_docre_bionlp2024/config/ma_qa.conf
```

Use `ma_atlop.conf` for MA-ATLOP and `ma_qa.conf` for MAQA.

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Then run:

```bash
bash ./run_ma_docre_train_eval.sh
```
