# KAPipe

KAPipe is a learnable pipeline for knowledge acquisition, with a particular focus on (semi-)automatically complementing knowledge bases in specialized domains.

## Features

- KAPipe provides **trained pipelines** for end-to-end knowledge graph construction from text.
- A pipeline is designed as a cascade of the following task components:
    - **Named Entity Recognition (NER):** Extracting entity mention spans and their entity types from the input text.
    - **Entity Disambiguation - Retrieval (ED-Retrieval)**: Retrieving a set of candidate entity IDs for each given mention in the text, based on a knowledge-base entity pool.
    - **Entity Disambiguation - Reranking (ED-Reranking)**: Reranking the retrieved entity IDs and selecting the most likely entity ID for each given mention in the text.
    - **Document-level Relation Extraction (DocRE)**: Extracting a set of relational triples (head entity, relation, tail entity) for a given entity set.
- It is possible to use only specific task components.
- KAPipe uses the **state-of-the-art models** for each task component.
- KAPipe also supports **training** of the pipeline (or specific task components) for new domains, entity types, relation labels, and knowledge bases.
- This repository also contains the source codes for experiments on custom models, including BERT-based supervised learning models and Large Language Model (LLM)-based In-Context Learning. The following customizable models are implemented for each task:
    - NER: Biaffine-NER ([`Yu et al., 2020`](https://aclanthology.org/2020.acl-main.577/)), LLM-NER
    - ED-Retrieval: BLINK Bi-Encoder ([`Wu et al., 2020`](https://aclanthology.org/2020.emnlp-main.519/)), BM25, Levenshtein-based retriever
    - ED-Reranking: BLINK Cross-Encoder ([`Wu et al., 2020`](https://aclanthology.org/2020.emnlp-main.519/)), LLM-ED
    - DocRE: ATLOP ([`Zhou et al., 2021`](https://ojs.aaai.org/index.php/AAAI/article/view/17717)), LLM-DocRE, MA-ATLOP and MA-QA (Oumaima and Nishida et al., 2024)

## Installation

```bash
python -m venv .env
source .env/bin/activate
pip install -U pip setuptools wheel
pip install kapipe
```

## Data Format: *Document*

We define a common dictionary format as the input and output for the pipeline.
We call this dictionary format **"Document"**.
The pipeline is a cascade of the tasks components, and the input and output of each task component is also a Document.
The information in the input Document is either passed on to the output Document or updated.
Note: As Documents are just dictionary data, users can add their own meta-information, such as information on the correspondence between each word and its position on the PDF, to the Documents, and this information will be retained in the pipeline's output.

### Input:

```json
{
    "doc_key": "6794356",
    "sentences": [
        "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant .",
        "A newborn with massive tricuspid regurgitation , atrial flutter , congestive heart failure , and a high serum lithium level is described .",
        "This is the first patient to initially manifest tricuspid regurgitation and atrial flutter , and the 11th described patient with cardiac disease among infants exposed to lithium compounds in the first trimester of pregnancy .",
        "Sixty - three percent of these infants had tricuspid valve involvement .",
        "Lithium carbonate may be a factor in the increasing incidence of congenital heart disease when taken during early pregnancy .",
        "It also causes neurologic depression , cyanosis , and cardiac arrhythmia when consumed prior to delivery ."
    ]
}
```

### Output:

```json
{
    "doc_key": "6794356",
    "sentences": [
        "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant .",
        "A newborn with massive tricuspid regurgitation , atrial flutter , congestive heart failure , and a high serum lithium level is described .",
        "This is the first patient to initially manifest tricuspid regurgitation and atrial flutter , and the 11th described patient with cardiac disease among infants exposed to lithium compounds in the first trimester of pregnancy .",
        "Sixty - three percent of these infants had tricuspid valve involvement .",
        "Lithium carbonate may be a factor in the increasing incidence of congenital heart disease when taken during early pregnancy .",
        "It also causes neurologic depression , cyanosis , and cardiac arrhythmia when consumed prior to delivery ."
    ],
    "mentions": [
        {
            "span": [
                0,
                2
            ],
            "name": "Tricuspid valve regurgitation",
            "entity_type": "Disease",
            "entity_id": "D014262"
        },
        {
            "span": [
                4,
                5
            ],
            "name": "lithium carbonate",
            "entity_type": "Chemical",
            "entity_id": "D016651"
        },
        {
            "span": [
                6,
                6
            ],
            "name": "toxicity",
            "entity_type": "Disease",
            "entity_id": "D064420"
        },
        {
            "span": [
                16,
                17
            ],
            "name": "tricuspid regurgitation",
            "entity_type": "Disease",
            "entity_id": "D014262"
        },
        {
            "span": [
                19,
                20
            ],
            "name": "atrial flutter",
            "entity_type": "Disease",
            "entity_id": "D001282"
        },
        {
            "span": [
                22,
                24
            ],
            "name": "congestive heart failure",
            "entity_type": "Disease",
            "entity_id": "D006333"
        },
        {
            "span": [
                30,
                30
            ],
            "name": "lithium",
            "entity_type": "Chemical",
            "entity_id": "D008094"
        },
        {
            "span": [
                43,
                44
            ],
            "name": "tricuspid regurgitation",
            "entity_type": "Disease",
            "entity_id": "D014262"
        },
        {
            "span": [
                46,
                47
            ],
            "name": "atrial flutter",
            "entity_type": "Disease",
            "entity_id": "D001282"
        },
        {
            "span": [
                55,
                56
            ],
            "name": "cardiac disease",
            "entity_type": "Disease",
            "entity_id": "D006331"
        },
        {
            "span": [
                61,
                61
            ],
            "name": "lithium",
            "entity_type": "Chemical",
            "entity_id": "D008094"
        },
        {
            "span": [
                82,
                83
            ],
            "name": "Lithium carbonate",
            "entity_type": "Chemical",
            "entity_id": "D016651"
        },
        {
            "span": [
                93,
                95
            ],
            "name": "congenital heart disease",
            "entity_type": "Disease",
            "entity_id": "D006331"
        },
        {
            "span": [
                105,
                106
            ],
            "name": "neurologic depression",
            "entity_type": "Disease",
            "entity_id": "D003866"
        },
        {
            "span": [
                108,
                108
            ],
            "name": "cyanosis",
            "entity_type": "Disease",
            "entity_id": "D003490"
        },
        {
            "span": [
                111,
                112
            ],
            "name": "cardiac arrhythmia",
            "entity_type": "Disease",
            "entity_id": "D001145"
        }
    ],
    "entities": [
        {
            "mention_indices": [
                0,
                3,
                7
            ],
            "entity_type": "Disease",
            "entity_id": "D014262"
        },
        {
            "mention_indices": [
                1,
                11
            ],
            "entity_type": "Chemical",
            "entity_id": "D016651"
        },
        {
            "mention_indices": [
                2
            ],
            "entity_type": "Disease",
            "entity_id": "D064420"
        },
        {
            "mention_indices": [
                4,
                8
            ],
            "entity_type": "Disease",
            "entity_id": "D001282"
        },
        {
            "mention_indices": [
                5
            ],
            "entity_type": "Disease",
            "entity_id": "D006333"
        },
        {
            "mention_indices": [
                6,
                10
            ],
            "entity_type": "Chemical",
            "entity_id": "D008094"
        },
        {
            "mention_indices": [
                9,
                12
            ],
            "entity_type": "Disease",
            "entity_id": "D006331"
        },
        {
            "mention_indices": [
                13
            ],
            "entity_type": "Disease",
            "entity_id": "D003866"
        },
        {
            "mention_indices": [
                14
            ],
            "entity_type": "Disease",
            "entity_id": "D003490"
        },
        {
            "mention_indices": [
                15
            ],
            "entity_type": "Disease",
            "entity_id": "D001145"
        }
    ],
    "relations": [
        {
            "arg1": 1,
            "relation": "CID",
            "arg2": 7
        },
        {
            "arg1": 1,
            "relation": "CID",
            "arg2": 8
        },
        {
            "arg1": 1,
            "relation": "CID",
            "arg2": 9
        }
    ]
}
```

## Downloading Trained Pipelines

Trained pipelines can be downloaded from https://drive.google.com/drive/folders/16ypMCoLYf5kDxglDD_NYoCNAfhTy4Qwp.

Download the latest compressed file `release.YYYYMMDD.tar.gz`, and then unzip it in `~/.kapipe` directory as follows:

```bash
mkdir ~/.kapipe
mv release.YYYYMMDD.tar.gz ~/.kapipe
cd ~/.kapipe
tar -zxvf release.YYYYMMDD.tar.gz
```

## Loading and Using Pipeline

The easiest way to apply the knowledge acquisition pipeline (i.e., the cascade of NER, ED, and DocRE tasks) to an input document is to load the pipeline using `kapipe.load()` and just apply it to the document.

```python
import kapipe
ka = kapipe.load("cdr_biaffinener_blink_atlop")
document = ka(document)
```

The above code loads and uses models that have already been trained for specific domains, entity types (in NER), knowledge bases (in ED), and relation labels (in DocRE).
Specifically, the identifier `"cdr_biaffinener_blink_atlop"` above indicates that the Biaffine-NER, BLINK, and ATLOP models trained on the CDR dataset (biomedical abstracts, Chemical and Disease entity types, entity IDs based on the MeSH ontology, and Chemical-Induce-Disease relation label) are used for NER, ED, and DocRE, respectively.

It is also possible to apply specific tasks by directly calling the task components.
For example, if you would like to perform only NER and ED, please do the following.

```python
import kapipe
ka = kapipe.load("cdr_biaffinener_blink_atlop")

# NER
document = ka.ner(document)
# ED-Retrieval
document, candidate_entities = ka.ed_ret(document, num_candidate_entities=10)
# ED-Reranking
document = ka.ed_rank(document, candidate_entities)
```

Also, for example, if mentions and entities have already been annotated (by humans or external systems), and if you would like to perform only DocRE, do the following.
Note that the mentions and entities have already been integrated into the input document.

```python
import kapipe
ka = kapipe.load("cdr_biaffinener_blink_atlop")

# DocRE
document = ka.docre(document_with_gold_mentions_and_entities)
```

## Available Trained Pipelines

The following pipelines are currently available.

| identifier | NER Model and Dataset (Entity Types) | ED-Retrieval Model and Dataset with Knowledge Base | ED-Reranking Model and Dataset with Knowledge Base | DocRE Model and Dataset (Relation Labels) |
| --- | --- | --- | --- | --- |
| cdr_biaffinener_blink_atlop | Biaffine-NER on CDR (Chemical, Disease) | BLINK Bi-Encoder on CDR + MeSH (2015) | BLINK Cross-Encoder on CDR + MeSH (2015) | ATLOP on CDR (Chemical-Induce-Disease) |

## Training

If the trained pipelines do not cover your target domain, entity types, knowledge base, or relation labels, please train each task component in the pipeline on your dataset.
Once you have trained the pipeline, please save it for future reuse. You can set your own identifier.

```python
import kapipe

ka = kapipe.blank(gpu_map={"ner":0, "ed_retrieval":1, "ed_reranking": 2, "docre": 3})

# NER
ka.ner.fit(
    train_documents,
    dev_documents,
    optional_config={
        "bert_pretrained_name_or_path": "allenai/scibert_scivocab_uncased",
        "bert_learning_rate": 2e-5,
        "task_learning_rate": 1e-4,
        "dataset_name": "example_dataset",
        "allow_nested_entities": True,
        "max_epoch": 10,
    }
)

# ED-Retrieval
ka.ed_ret.fit(
    entity_dict,
    train_documents,
    dev_documents,
    optional_config={
        "bert_pretrained_name_or_path": "allenai/scibert_scivocab_uncased",
        "bert_learning_rate": 2e-5,
        "task_learning_rate": 1e-4,
        "dataset_name": "example_dataset",
        "max_epoch": 10,
    }
)

# ED-Reranking
train_candidate_entities = [
	ka.ed_ret(d, retrieval_size=128)[1] for d in train_documents
]
dev_candidate_entities = [
	ka.ed_ret(d, retrieval_size=128)[1] for d in dev_documents
]
ka.ed_rank.fit(
    entity_dict,
    train_documents, train_candidate_entities,
    dev_documents, dev_candidate_entities,
    optional_config={
        "bert_pretrained_name_or_path": "allenai/scibert_scivocab_uncased",
        "bert_learning_rate": 2e-5,
        "task_learning_rate": 1e-4,
        "dataset_name": "example_dataset",
        "max_epoch": 10,
    }
)

# DocRE
ka.docre.fit(
    train_documents,
    dev_documents,
    optional_config={
        "bert_pretrained_name_or_path": "allenai/scibert_scivocab_uncased",
        "bert_learning_rate": 2e-5,
        "task_learning_rate": 1e-4,
        "dataset_name": "example_dataset",
        "max_epoch": 10,
        "possible_head_entity_types": ["Chemical"], # or None
        "possible_tail_entity_types": ["Disease"], # or None
    }
)
 
ka.save("your favorite identifier")
```

If you would like to train only specific task components (for example, if you would like to use models trained on CDR for NER and DocRE, and train a new model on a different version of MeSH for ED), please do the following.

```python
import kapipe

ka = kapipe.load("cdr_biaffinener_blink_atlop")

# ED-Retrieval
ka.ed_ret.fit(
    entity_dict,
    train_documents,
    dev_documents
)

# ED-Reranking
train_candidate_entities = [
	ka.ed_ret(d, retrieval_size=128)[1] for d in train_documents
]
dev_candidate_entities = [
	ka.ed_ret(d, retrieval_size=128)[1] for d in dev_documents
]
ka.ed_rank.fit(
    entity_dict,
    train_documents, train_candidate_entities,
    dev_documents, dev_candidate_entities
)

ka.save("your favorite identifier")
```

When using the saved pipeline, please load it by specifying the identifier.

```python
import kapipe

kapipe.load("your favorite identifier")
```

## Experiments on Custom Models

The pipeline is a top-level wrapper class that consists of a cascade of task components, and each task component is also a black box class, in which specific models (e.g., Biaffine-NER, ATLOP) are used.
In order to perform various training, evaluation, and analysis on specific methods, it may be more intuitive to directly instantiate each method (hereafter referred to as a "system") rather than the pipeline.

The core of KAPipe is the systems, and the pipeline is just a wrapper to make them easy to use with minimal coding. If you are familiar with coding and your goal is not just to apply the KA pipeline, but also to develop the methods, it would be better to work directly with the systems rather than using the pipeline.

The fastest way to find out how to initialize, train and evaluate each system is to look at the `experiments/codes/run_*` scripts.