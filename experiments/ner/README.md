# NER

This directory contains example experiments for NER.

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/ner
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
experiments/ner/data/examples/documents.json
experiments/ner/data/examples/documents_with_supervision.json
experiments/ner/data/examples/demonstration_documents.json
```

### Benchmark datasets

If necessary, you can prepare benchmark datasets with the following commands.

```bash
cd experiments/ner/dataset-preparation

bash ./prepare_conll2003.sh
bash ./prepare_linked_docred.sh
bash ./prepare_cdr.sh
bash ./prepare_medmentions.sh
```

These commands extract each dataset under `experiments/ner/data/`.

Example:
```bash
experiments/ner/data/cdr/train.json
experiments/ner/data/cdr/dev.json
experiments/ner/data/cdr/test.json
experiments/ner/data/cdr/demonstration_documents.json
```

## Step 3. Running the NER Component

First, check `STORAGE` in each execution script and adjust it to your environment.

To apply an off-the-shelf NER component to documents, run:

```bash
bash ./run_ner.sh
```

To train or evaluate an NER component, run:

```bash
bash ./run_ner_train_eval.sh
```

By default, both scripts use the LLM-NER method (`method_name = llm_ner`).
If you use the OpenAI API, set your API key in advance.

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

To use Biaffine-NER (`method_name = biaffine_ner`), change `METHOD` in each script to `biaffine_ner`.