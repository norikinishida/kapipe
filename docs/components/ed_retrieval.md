# Entity Disambiguation (Retrieval) (`kapipe.ed_retrieval`)

**Entity Disambiguation (Retrieval)** retrieves candidate concept IDs from a knowledge base for each entity mention in a document.

This component is the candidate retrieval step of entity disambiguation. The retrieved candidates can be passed to an entity disambiguation reranking component.

## Input

A document is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `doc_key` | `str` | Unique document identifier |
| `sentences` | `list[str]` | Tokenized sentences |
| `mentions` | `list[dict]` | Entity mentions |

Each mention contains the following fields.

| Field | Type | Description |
|---|---|---|
| `span` | `tuple[int, int]` | Token-based mention span |
| `name` | `str` | Mention string |
| `entity_type` | `str` | Entity type |

```json
{
    "doc_key": "6794356",
    "sentences": [
        "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant .",
        ...
    ],
    "mentions": [
        {
            "span": [0, 2],
            "name": "Tricuspid valve regurgitation",
            "entity_type": "Disease"
        },
        ...
    ]
}
```

## Output

The component returns two objects: an updated document and candidate entities for the document.

The output document preserves the input fields and adds `entity_id` to each mention and `entities` to the document.

| Field | Type | Description |
|---|---|---|
| `mentions` | `list[dict]` | Entity mentions with retrieved concept IDs |
| `entities` | `list[dict]` | Entities aggregated from mentions |

Each mention additionally contains the following field.

| Field | Type | Description |
|---|---|---|
| `entity_id` | `str` | Concept ID |

Each entity contains the following fields.

| Field | Type | Description |
|---|---|---|
| `mention_indices` | `list[int]` | Indices of mentions belonging to this entity |
| `mention_names` | `list[str]` | Mention strings belonging to this entity |
| `entity_type` | `str` | Entity type |
| `entity_id` | `str` | Concept ID |

```json
{
    "doc_key": "6794356",
    "sentences": [
        "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant .",
        ...
    ],
    "mentions": [
        {
            "span": [0, 2],
            "name": "Tricuspid valve regurgitation",
            "entity_type": "Disease",
            "entity_id": "D014262"
        },
        ...
    ],
    "entities": [
        {
            "mention_indices": [0, 3, 7],
            "mention_names": [
                "Tricuspid valve regurgitation",
                "tricuspid regurgitation",
                "tricuspid regurgitation"
            ],
            "entity_type": "Disease",
            "entity_id": "D014262"
        },
        ...
    ],
 
}
```

Candidate entities are represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `doc_key` | `str` | Unique document identifier |
| `candidate_entities` | `list[list[dict]]` | Candidate entities for each mention |

The outer list of `candidate_entities` is aligned with `document["mentions"]`.

Each candidate entity contains the following fields.

| Field | Type | Description |
|---|---|---|
| `entity_id` | `str` | Candidate concept ID |
| `canonical_name` | `str` | Canonical entity name, if available |
| `score` | `float` | Retrieval score |

```json
{
    "doc_key": "6794356",
    "candidate_entities": [
        [
            {
                "entity_id": "D014262",
                "canonical_name": "Tricuspid Valve Insufficiency",
                "score": 0.0017849934520199895
            },
            {
                "entity_id": "D014264",
                "canonical_name": "Tricuspid Valve Stenosis",
                "score": 0.0017764709191396832
            },
            ...
        ],
        ...
    ]
}
```

## Supported Methods

| Method | Description |
|---|---|
| Mention-name Assignment | Assigns each mention surface form (lowercased) as its entity ID. This is useful when no explicit target knowledge base exists and mention surface forms are treated as pseudo concepts. |
| [BLINK Bi-Encoder (Wu et al., 2020)](https://aclanthology.org/2020.emnlp-main.519/) | Retrieves candidate entities from a predefined entity dictionary using dense bi-encoder retrieval. |

## Public Snapshots

The following public snapshots can be loaded with `from_identifier(...)`.

| Method | Identifier | Dataset | Entity Dictionary | Configuration |
|---|---|---|---|---|
| BLINK Bi-Encoder | `blink_bi_encoder_linked_docred` | Linked-DocRED | DBpedia 2020.02.01 | `bert-base-uncased`; precomputed entity vectors included |
| BLINK Bi-Encoder | `blink_bi_encoder_cdr` | CDR | MeSH 2015 | `allenai/scibert_scivocab_uncased`; precomputed entity vectors included |

These snapshots are predefined resources for existing benchmark settings. You can also use your own entity dictionary by training a retriever for it.

`identifier` is resolved through the public resource configuration installed under `~/.kapipe/download/config`.

## Usage

### Mention-name Assignment:

```python
from kapipe.ed_retrieval import MentionNameEntityRetriever

# Instantiate the ED-Retrieval component using simple mention-name assignment
retriever = MentionNameEntityRetriever()

# Assign mention surface forms as entity IDs
result_document, candidate_entities_for_doc = retriever.search(
    document=document,
    retrieval_size=1,
)
```

### Predefined BLINK Bi-Encoder ED-Retrieval:

```python
from kapipe.ed_retrieval import BlinkBiEncoder

# Load the BLINK Bi-Encoder ED-Retrieval component predefined for the (Linked-DocRED, DBPedia) schema
retriever = BlinkBiEncoder.from_identifier(
    identifier="blink_bi_encoder_linked_docred"
)

# Build an approximate nearest neighbor index over entities
retriever.make_index(use_precomputed_entity_vectors=True)

# Retrieve the top-10 candidate entities for each mention
result_document, candidate_entities_for_doc = retriever.search(
    document=document,
    retrieval_size=10,
)
```

## Training with a Custom Entity Dictionary (Concepts)

If you want to use your own entity dictionary with `BlinkBiEncoder`, train the BLINK Bi-Encoder model for that dictionary first.

See [experiments/ed_retrieval/run_ed_retrieval_trani_eval.py](../../experiments/ed_retrieval/run_ed_retrieval_train_eval.py) for runnable training and evaluation examples.

## Example

See [experiments/ed_retrieval](../../experiments/ed_retrieval) for runnable examples.