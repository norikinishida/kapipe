# Named Entity Recognition (`kapipe.ner`)

**Named Entity Recognition (NER)** extracts entity mentions from a document and assigns an entity type to each mention.

## Input

A document is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `doc_key` | `str` | Unique document identifier |
| `sentences` | `list[str]` | Tokenized sentences |

```json
{
    "doc_key": "6794356",
    "sentences": [
        "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant .",
        ...
    ]
}
```

## Output

The output document preserves the input fields and adds `mentions`.

| Field | Type | Description |
|---|---|---|
| `mentions` | `list[dict]` | Extracted entity mentions |

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

## Supported Methods

| Method | Description |
|---|---|
| [Biaffine-NER (Yu et al., 2020)](https://aclanthology.org/2020.acl-main.577/) | Span-based neural NER extractor based on Biaffine scoring |
| LLM-based NER | Prompt-based NER extractor using a proprietary or open-source LLM |

## Usage

### Predefined Biaffine-NER:

```python
from kapipe.ner import BiaffineNER

# Load Biaffine-NER predefined for the Linked-DocRED schema
extractor = BiaffineNER.from_identifier(identifier="biaffine_ner_linked_docred")

# Extract entity mentions from a document
result_document = extractor.extract(document)
```

### Predefined LLM-based NER:

```python
from kapipe.ner import LLMNER

# Instantiate your LLM wrapper
model = ...

# Load LLM-based NER predefined for the Linked-DocRED schema
extractor = LLMNER.from_identifier(
    model=model,
    identifier="llm_ner_linked_docred"
)

# Extract entity mentions from a document
result_document = extractor.extract(document)
```

### User-defined LLM-based NER:

```python
from kapipe.ner import LLMNER

# Instantiate your LLM wrapper
model = ...

# Define entity types for your task
vocab_etype = {
    "Person": 0,
    "Location": 1,
}

# Define metadata used in the prompt
etype_meta_info = {
    "Person": {
        "Pretty Name": "Person",
        "Definition": "A person, including real and fictional people.",
    },
    "Location": {
        "Pretty Name": "Location",
        "Definition": "A physical or geopolitical location.",
    },
}

# Build a user-defined LLM-based NER extractor
extractor = LLMNER(
    model=model,
    prompt_template_name_or_path="ner_13_zeroshot",
    vocab_etype=vocab_etype,
    etype_meta_info=etype_meta_info,
)


# Extract entity mentions from a document
result_document = extractor.extract(document=document)
```

## Training with a Custom Entity Type Schema

If you want to use your own entity type schema with `BiaffineNER`, train the BiaffineNER model for that schema first.

See [experiments/ner/run_ner_train_eval.py](../../experiments/ner/run_ner_train_eval.py) for runnable training and evaluation examples.

## Example

See [experiments/ner](../../experiments/ner) for runnable examples.
