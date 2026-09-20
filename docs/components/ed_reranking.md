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
    "doc_key": "8800187",
    "candidate_entities": [
        [
            {
                "entity_id": "D002122",
                "canonical_name": "Calcium Chloride",
                "score": 0.0017943419516086578
            },
            {
                "entity_id": "D002118",
                "canonical_name": "Calcium",
                "score": 0.0017746267840266228
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

# Instantiate the ED-Reranking component using identical function
reranker = IdenticalEntityReranker()

# Keep the retrieved concept IDs unchanged
result_document = reranker.rerank(
    document=document,
    candidate_entities_for_doc=candidate_entities_for_doc,
)
```

### Predefined BLINK Cross-Encoder ED-Reranking:

```python
from kapipe.ed_reranking import BlinkCrossEncoder

# Load the BLINK Cross-Encoder ED-Reranking component predefined for the (Linked-DocRED, DBPedia) schema
reranker = BlinkCrossEncoder.from_identifier(
    identifier="blink_cross_encoder_linked_docred"
)

# Rerank candidate entities for each mention
result_document = reranker.rerank(
    document=document,
    candidate_entities_for_doc=candidate_entities_for_doc,
)
```

### Predefined LLM-based ED-Reranking:

```python
from kapipe.ed_reranking import LLMED

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

# Load the LLM-based ED-Reranking component predefined for the (Linked-DocRED, DBPedia) schema
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

### User-defined LLM-based ED-Reranking:

```python
from kapipe.ed_reranking import LLMED

# Instantiate your LLM wrapper
model = ...

# Instantiate the LLM-based ED-Reranking component with the user-defined entity dictionary
reranker = LLMED(
    model=model,
    prompt_template_name_or_path="ed_11_zeroshot",
    knowledge_base_name="<your KB name>",
    entity_dict_path="/path/to/entity_dict.json",
)

# Rerank candidate entities for each mention
result_document = reranker.rerank(
    document=document,
    candidate_entities_for_doc=candidate_entities_for_doc,
)
```

## OpenAI Batch API

`LLMED` supports the OpenAI Batch API when its model is an `OpenAILLM` instance.

```python
# Submit all ED-reranking prompts to one or more OpenAI batches
batch_ids: list[str] = reranker.submit_batch(
    documents=documents,
    candidate_entities=candidate_entities,
)

# Fetch and process the results after all OpenAI batches are complete
result_documents = reranker.fetch_and_process_batch(
    documents=documents,
    candidate_entities=candidate_entities,
    batch_ids=batch_ids,
)
```

`submit_batch()` automatically splits requests into batches containing at most 50,000 requests and 200 MB of JSONL input. It returns the batch IDs in submission order, and `fetch_and_process_batch()` merges their responses in the original document order. Mentions are grouped into requests of up to five per document; documents without mentions do not produce Batch API requests. If every document has no mentions, `submit_batch()` raises a `ValueError`.

Pass the same `documents` and `candidate_entities` in the same order to both methods. The two lists must have the same length, and corresponding `doc_key` values must match. Keep the model settings and prompt template unchanged between submission and fetching. `fetch_and_process_batch()` raises a `RuntimeError` if any OpenAI batch is incomplete or contains failed requests.

## Training with a Custom Entity Dictionary (Concepts)

If you want to use your own entity dictionary with `BlinkCrossEncoder`, train the BLINK Cross-Encoder model for that dictionary first.

See [experiments/ed_reranking/run_ed_reranking_train_eval.py](../../experiments/ed_reranking/run_ed_reranking_train_eval.py) for runnable training and evaluation examples.

## Custom Prompt Templates

`prompt_template_name_or_path` accepts either the name of a built-in prompt template ([`kapipe/ed_reranking/prompt_templates/*.txt`](../../kapipe/ed_reranking/prompt_templates)) or a path to a user-defined prompt template.

A custom prompt template for `LLMED` supports the following placeholders.

| Placeholder | Required | Description |
|---|---|---|
| `{knowledge_base_name_prompt}` | Yes | Name of the knowledge base containing the candidate entities |
| `{demonstrations_prompt}` | No | Formatted demonstration documents, candidate entities, and gold entity assignments |
| `{test_case_prompt}` | Yes | Current document, target mentions, and candidate entities to rerank |

`{demonstrations_prompt}` is used by few-shot templates. It can be omitted from zero-shot templates.

The component validates that `{knowledge_base_name_prompt}` and `{test_case_prompt}` are present.

## Example

See [experiments/ed_reranking](../../experiments/ed_reranking) for runnable examples.
