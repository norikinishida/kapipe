# Entity Disambiguation (Reranking) (`kapipe.ed_reranking`)

**Entity Disambiguation (Reranking)** selects the most likely concept ID for each entity mention from retrieved candidate entities.

This component is the reranking step of entity disambiguation. It takes a document and candidate entities produced by an entity disambiguation retrieval component.

## Input

The input consists of a document and candidate entities for the document.

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
| `entity_id` | `str` | Concept ID assigned by the retrieval component |

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
    ]
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

## Output

The output document preserves the input fields and updates `entity_id` for each mention. It also adds (or updates) `entities` to the document.

| Field | Type | Description |
|---|---|---|
| `mentions` | `list[dict]` | Entity mentions with reranked concept IDs |
| `entities` | `list[dict]` | Entities aggregated from mentions |

Each mention contains the following fields.

| Field | Type | Description |
|---|---|---|
| `span` | `tuple[int, int]` | Token-based mention span |
| `name` | `str` | Mention string |
| `entity_type` | `str` | Entity type |
| `entity_id` | `str` | Reranked concept ID |

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

## Supported Methods

| Method | Description |
|---|---|
| Identical Entity Reranker | Keeps the concept IDs assigned by the retrieval component unchanged. This is useful when no reranking is needed or when the retrieval result is used as the final disambiguation result. |
| [BLINK Cross-Encoder (Wu et al., 2020)](https://aclanthology.org/2020.emnlp-main.519/) | Reranks candidate entities using a cross-encoder over mention contexts and entity descriptions. |
| LLM-based ED | Reranks candidate entities using a proprietary or open-source LLM with an entity disambiguation prompt. |

## Public Snapshots

The following public snapshots can be loaded with `from_identifier(...)`.

| Method | Identifier | Dataset | Entity Dictionary | Configuration |
|---|---|---|---|---|
| BLINK Cross-Encoder | `blink_cross_encoder_linked_docred` | Linked-DocRED | DBpedia 2020.02.01 | `bert-base-uncased`; up to 16 candidates per mention at inference |
| BLINK Cross-Encoder | `blink_cross_encoder_cdr` | CDR | MeSH 2015 | `allenai/scibert_scivocab_uncased`; up to 16 candidates per mention at inference |
| LLM-based ED | `llm_ed_linked_docred` | Linked-DocRED | DBpedia 2020.02.01 | Few-shot prompt snapshot; runtime LLM is user-provided |
| LLM-based ED | `llm_ed_cdr` | CDR | MeSH 2015 | Few-shot prompt snapshot; runtime LLM is user-provided |

These snapshots are predefined resources for existing benchmark settings. You can also use your own entity dictionary by training or configuring a reranker for it.

`identifier` is resolved through the public resource configuration installed under `~/.kapipe/download/config`.

## Usage

### Identical Entity Reranking:

```python
from kapipe.ed_reranking import IdenticalEntityReranker

# Build an Identical Entity Reranker
reranker = IdenticalEntityReranker()

# Keep the retrieved concept IDs unchanged
result_document = reranker.rerank(
    document=document,
    candidate_entities_for_doc=candidate_entities_for_doc,
)
```

### Predefined BLINK Cross-Encoder:

```python
from kapipe.ed_reranking import BlinkCrossEncoder

# Load a BLINK Cross-Encoder component predefined for the (Linked-DocRED, DBPedia) schema
reranker = BlinkCrossEncoder.from_identifier(
    identifier="blink_cross_encoder_linked_docred"
)

# Rerank candidate entities for each mention
result_document = reranker.rerank(
    document=document,
    candidate_entities_for_doc=candidate_entities_for_doc,
)
```

### Predefined LLM-based ED:

```python
from kapipe.ed_reranking import LLMED

# Instantiate your LLM wrapper
# OpenAI LLM
from kapipe.llms import OpenAILLM
model = OpenAILLM(
    model_name="gpt-5.4-nano",
    max_new_tokens=1024,
)
# HuggingFace LLM
from kapipe.llms import HuggingFaceLLM
model = HuggingFaceLLM(
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    max_new_tokens=1024,
    quantization_bits=4,
)

# Load an LLM-based ED component predefined for the (Linked-DocRED, DBPedia) schema
reranker = LLMED.from_identifier(
    model=model,
    identifier="llm_ed_linked_docred"
)

# Rerank candidate entities for each mention
result_document = reranker.rerank(
    document=document,
    candidate_entities_for_doc=candidate_entities_for_doc,
)
```

### User-defined LLM-based ED:

```python
from kapipe.ed_reranking import LLMED

# Instantiate your LLM wrapper
model = ...

# Instantiate a user-defined LLM-based ED component
reranker = LLMED(
    model=model,
    prompt_template_name_or_path="ed_04_zeroshot",
    knowledge_base_name="<your KB name>",
    entity_dict_path="/path/to/entity_dict.json",
)

# Rerank candidate entities for each mention
result_document = reranker.rerank(
    document=document,
    candidate_entities_for_doc=candidate_entities_for_doc,
)
```

## Training with a Custom Entity Dictionary (Concepts)

If you want to use your own entity dictionary with `BlinkCrossEncoder`, train the BLINK Cross-Encoder model for that dictionary first.

See [experiments/ed_reranking/run_ed_reranking_train_eval.py](../../experiments/ed_reranking/run_ed_reranking_train_eval.py) for runnable training and evaluation examples.

## Example

See [experiments/ed_reranking](../../experiments/ed_reranking) for runnable examples.