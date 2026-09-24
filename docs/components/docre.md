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
    "doc_key": "8800187",
    "sentences": [
        "Effect of calcium chloride and 4 - aminopyridine therapy on desipramine toxicity in rats .",
        ...
    ],
    "mentions": [
        {
            "span": [2, 3],
            "name": "calcium chloride",
            "entity_type": "Chemical",
            "entity_id": "D002122"
        },
        ...
    ],
    "entities": [
        {
            "mention_indices": [0, 11, 16, 22, 26, 27, 30],
            "mention_names": [
                "calcium chloride",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2"
            ],
            "entity_type": "Chemical",
            "entity_id": "D002122"
        },
        ...
    ]
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
    "doc_key": "8800187",
    "sentences": [
        "Effect of calcium chloride and 4 - aminopyridine therapy on desipramine toxicity in rats .",
        ...
    ],
    "mentions": [
        {
            "span": [2, 3],
            "name": "calcium chloride",
            "entity_type": "Chemical",
            "entity_id": "D002122"
        },
        ...
    ],
    "entities": [
        {
            "mention_indices": [0, 11, 16, 22, 26, 27, 30],
            "mention_names": [
                "calcium chloride",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2"
            ],
            "entity_type": "Chemical",
            "entity_id": "D002122"
        },
        ...
    ],
    "relations": [
        {
            "arg1": 0,
            "relation": "CID",
            "arg2": 9
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
    max_new_tokens=8192,
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
    prompt_template_name_or_path="docre_10_zeroshot",
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

## OpenAI Batch API

`LLMDocRE` supports the OpenAI Batch API when its model is an `OpenAILLM` instance.

```python
# Submit all document prompts to one or more OpenAI batches
batch_ids: list[str] = extractor.submit_batch(documents=documents)

# Fetch and process the results after all OpenAI batches are complete
result_documents = extractor.fetch_and_process_batch(
    documents=documents,
    batch_ids=batch_ids,
)
```

`submit_batch()` automatically splits requests into batches containing at most 50,000 requests and 200 MB of JSONL input. It returns the batch IDs in submission order, and `fetch_and_process_batch()` merges their responses in the original document order. Documents with one or fewer entities do not produce Batch API requests. If every document has one or fewer entities, `submit_batch()` raises a `ValueError`.

Pass the same `documents` in the same order to both methods. Keep the model settings and prompt template unchanged between submission and fetching. `fetch_and_process_batch()` raises a `RuntimeError` if any OpenAI batch is incomplete or contains failed requests.

## Training with a Custom Relation Schema

If you want to use your own relation schema with `ATLOP`, `MAATLOP`, or `MAQA`, train the DocRE model for that schema first.

See [experiments/docre/run_docre_train_eval.py](../../experiments/docre/run_docre_train_eval.py) for runnable training and evaluation examples.

## Custom Prompt Templates

`prompt_template_name_or_path` accepts either the name of a built-in prompt template ([`kapipe/docre/prompt_templates/*.txt`](../../kapipe/docre/prompt_templates)) or a path to a user-defined prompt template.

A custom prompt template for `LLMDocRE` supports the following placeholders.

| Placeholder | Required | Description |
|---|---|---|
| `{knowledge_base_name_prompt}` | Yes | Name of the knowledge base containing the entities |
| `{relations_prompt}` | Yes | Relation types and their descriptions, generated from `vocab_relation` and `rel_meta_info` |
| `{demonstrations_prompt}` | No | Formatted demonstration documents and their gold relations |
| `{test_case_prompt}` | Yes | Current document and its entities from which relations are extracted |

`{demonstrations_prompt}` is used by few-shot templates. It can be omitted from zero-shot templates.

The component validates that `{knowledge_base_name_prompt}`, `{relations_prompt}`, and `{test_case_prompt}` are present.

## Example

See [experiments/docre](../../experiments/docre) for runnable examples.
