# Document-level Relation Extraction (`kapipe.docre`)

**Document-level Relation Extraction (DocRE)** extracts relational triples between entities in a document.

This component takes a document with disambiguated entities and adds relations between entity pairs.

## Input

A document is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `doc_key` | `str` | Unique document identifier |
| `sentences` | `list[str]` | Tokenized sentences |
| `mentions` | `list[dict]` | Entity mentions |
| `entities` | `list[dict]` | Entities aggregated from mentions |

Each mention contains the following fields.

| Field | Type | Description |
|---|---|---|
| `span` | `tuple[int, int]` | Token-based mention span |
| `name` | `str` | Mention string |
| `entity_type` | `str` | Entity type |
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

## Output

The output document preserves the input fields and adds `relations`.

| Field | Type | Description |
|---|---|---|
| `relations` | `list[dict]` | Extracted relations between entities |

Each relation contains the following fields.

| Field | Type | Description |
|---|---|---|
| `arg1` | `int` | Index of the head entity |
| `relation` | `str` | Relation type |
| `arg2` | `int` | Index of the tail entity |

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
    "relations": [
        {
            "arg1": 1,
            "relation": "CID",
            "arg2": 7
        },
        ...
    ]
}
```

## Supported Methods

| Method | Description |
|---|---|
| [ATLOP (Zhou et al., 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17717) | Neural DocRE extractor based on adaptive thresholding and localized context pooling |
| [MA-ATLOP (Oumaima and Nishida et al., 2024)](https://aclanthology.org/2024.bionlp-1.37/) | Mention-agnostic extension of ATLOP |
| [MAQA (Oumaima and Nishida et al., 2024)](https://aclanthology.org/2024.bionlp-1.37/) | Mention-agnostic QA-based DocRE extractor |
| LLM-based DocRE | Prompt-based DocRE extractor using a proprietary or open-source LLM |

## Public Snapshots

The following public snapshots can be loaded with `from_identifier(...)`.

| Method | Identifier | Dataset | Relation Schema | Configuration |
|---|---|---|---|---|
| ATLOP | `atlop_linked_docred` | Linked-DocRED | 96 relations, e.g., `P17` (country), `P19` (place of birth), `P26` (spouse) | `bert-base-cased`; all entity-type pairs considered |
| ATLOP | `atlop_cdr` | CDR | 1 relation: `CID` (Chemical-Induce-Disease) | `allenai/scibert_scivocab_cased`; `Chemical` to `Disease` pairs only; overlap token embedding |
| LLM-based DocRE | `llm_docre_linked_docred` | Linked-DocRED | 96 relations, e.g., `P17` (country), `P19` (place of birth), `P26` (spouse) | Few-shot prompt snapshot; Wikipedia setting; runtime LLM is user-provided |
| LLM-based DocRE | `llm_docre_cdr` | CDR | 1 relation: `CID` (Chemical-Induce-Disease) | Few-shot prompt snapshot; MeSH setting; runtime LLM is user-provided |

These snapshots are predefined resources for common benchmark settings. You can also define your own relation schema and train or configure an extractor for it.

`identifier` is resolved through the public resource configuration installed under `~/.kapipe/download/config`.

## Usage

### Predefined ATLOP-based DocRE:

```python
from kapipe.docre import ATLOP

# Load the ATLOP-based DocRE component predefined for the Linked-DocRED schema
extractor = ATLOP.from_identifier(
    identifier="atlop_linked_docred"
)

# Extract relations from a document
result_document = extractor.extract(document=document)
```

### Predefined LLM-based DocRE:

```python
from kapipe.docre import LLMDocRE

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

# Load the LLM-based DocRE component predefined for the Linked-DocRED schema
extractor = LLMDocRE.from_identifier(
    model=model,
    identifier="llm_docre_linked_docred"
)

# Extract relations from a document
result_document = extractor.extract(document=document)
```

### User-defined LLM-based DocRE:

```python
from kapipe.docre import LLMDocRE

# Instantiate your LLM wrapper
model = ...

# Define relation types for your task
vocab_relation = {
    "works_for": 0,
    "located_in": 1,
}

# Define metadata used in the prompt
rel_meta_info = {
    "works_for": {
        "Pretty Name": "Works-For",
        "Definition": "The subject person works for the object organization.",
    },
    "located_in": {
        "Pretty Name": "Located-In",
        "Definition": "The subject entity is located in the object location.",
    },
}

# Instantiate the LLM-based DocRE component with the user-defined relation schema
extractor = LLMDocRE(
    model=model,
    prompt_template_name_or_path="docre_08_zeroshot",
    knowledge_base_name="<your KB name>",
    mention_style="all_mentions",
    with_span_annotation=True,
    possible_head_entity_types=None,
    possible_tail_entity_types=None,
    vocab_relation=vocab_relation,
    rel_meta_info=rel_meta_info,
)

# Extract relations from a document
result_document = extractor.extract(document=document)
```

## Training with a Custom Relation Schema

If you want to use your own relation schema with `ATLOP`, `MAATLOP`, or `MAQA`, train the DocRE model for that schema first.

See [experiments/docre/run_docre_train_eval.py](../../experiments/docre/run_docre_train_eval.py) for runnable training and evaluation examples.

## Example

See [experiments/docre](../../experiments/docre) for runnable examples.