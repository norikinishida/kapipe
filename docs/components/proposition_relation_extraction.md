# Proposition Relation Extraction (`kapipe.proposition_relation_extraction`)

**Proposition Relation Extraction** identifies directed logical and semantic relations between propositions.

This component (1) retrieves candidate tail propositions for each head proposition and (2) extract relations.

## Input

The input is a list of propositions.

Each proposition is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `passage_key` | `str` | Unique proposition identifier |
| `text` | `str` | Proposition text |
| `timestamp` | `str` | Date in `YYYY-MM-DD` format, if temporal relations are used |

Additional metadata fields are preserved during candidate retrieval and relation extraction.

```json
[
    {
        "passage_key": "proposition#001",
        "text": "An independent audit found that the Northbridge payment system processed 99.9% of transactions within two seconds in February 2025.",
        "timestamp": "2025-03-01"
    },
    {
        "passage_key": "proposition#002",
        "text": "A preliminary report found that the Northbridge payment system processed 97.0% of transactions within two seconds in January 2025.",
        "timestamp": "2025-02-01"
    },
    ...
]
```

## Output

The output is a list of directed proposition relation records.

Each relation record contains the head proposition, relation label, tail proposition, and explanation.

| Field | Type | Description |
|---|---|---|
| `head` | `dict` | Head proposition |
| `relation` | `str` | Directed relation from the head proposition to the tail proposition |
| `tail` | `dict` | Tail proposition |
| `explanation` | `str` | Explanation of the relation |

```json
[
    {
        "head": {
            "passage_key": "proposition#001",
            "text": "An independent audit found that the Northbridge payment system processed 99.9% of transactions within two seconds in February 2025.",
            "timestamp": "2025-03-01"
        },
        "relation": "updates",
        "tail": {
            "passage_key": "proposition#002",
            "text": "A preliminary report found that the Northbridge payment system processed 97.0% of transactions within two seconds in January 2025.",
            "timestamp": "2025-02-01"
        },
        "explanation": "The February audit updates the January processing rate from 97.0% to 99.9%."
    },
    ...
]
```

Entries classified as `NOREL` are excluded from the output.

## Relation Labels

The default prompt templates support the following relation labels.

| Label | Definition |
|---|---|
| `updates` | The head proposition provides newer information that replaces an older state or value in the tail proposition |
| `contradicts` | The head and tail propositions make incompatible factual claims that cannot both be true |
| `supports` | The head proposition provides evidence, reasons, or verification that increases the credibility of the tail proposition |
| `NOREL` | No direct logical relation applies; redundant statements and simple rephrasings without additional evidential value are also classified as `NOREL` |

## Supported Methods

| Method | Description |
|---|---|
| LLM-based Proposition Relation Extraction | Retrieves candidate proposition pairs and classifies their relations using a proprietary or open-source LLM |

## Usage

### LLM-based Proposition Relation Extraction:

```python
from kapipe.proposition_relation_extraction import LLMPropositionRelationExtractor

# Instantiate your LLM wrapper
# OpenAI LLM
from kapipe.llms import OpenAILLM
model = OpenAILLM(
    model_name="gpt-5.4-nano",
    max_new_tokens=2048,
)
# HuggingFace LLM
from kapipe.llms import HuggingFaceLLM
model = HuggingFaceLLM(
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    max_new_tokens=2048,
    quantization_bits=4,
)

# Instantiate a Passage Retrieval component for candidate retrieval
from kapipe.passage_retrieval import Contriever
retriever = Contriever(
    model_name="facebook/contriever-msmarco",
    max_passage_length=512,
    pooling_method="average",
    normalize=False,
    metric="inner-product",
)

# Instantiate the LLM-based Proposition Relation Extraction component
extractor = LLMPropositionRelationExtractor(
    model=model,
    retriever=retriever,
    prompt_template_name_or_path=(
        "proposition_relation_extraction_01_with_timestamp"
    ),
    use_timestamp=True,
)

# Build an index over propositions
extractor.make_index(
    propositions=propositions,
    index_dir="/path/to/index",
    batch_size=1024,
)

# Retrieve candidate tail propositions for a head proposition
tail_propositions = extractor.retrieve_tail_propositions(
    head_proposition=propositions[0],
    top_k=20,
    prefilter_k=100,
)

# Extract directed relations from the head proposition to candidate tails
triples = extractor.extract(
    head_proposition=propositions[0],
    tail_propositions=tail_propositions,
)
```

After building an index, you can reload it with `extractor.load_index(index_dir="/path/to/index")`.

## OpenAI Batch API

`LLMPropositionRelationExtractor` supports the OpenAI Batch API when its model is an `OpenAILLM` instance.

```python
# Retrieve candidate tail propositions for every head proposition
batch_tail_propositions = extractor.batch_retrieve_tail_propositions(
    head_propositions=propositions,
    top_k=20,
    prefilter_k=100,
    batch_size=1024,
)

# Submit all relation-extraction prompts to one or more OpenAI batches
batch_ids = extractor.submit_batch(
    head_propositions=propositions,
    batch_tail_propositions=batch_tail_propositions,
)

# Fetch and process the results after all OpenAI batches are complete
triples = extractor.fetch_and_process_batch(
    head_propositions=propositions,
    batch_tail_propositions=batch_tail_propositions,
    batch_ids=batch_ids,
)
```

`submit_batch()` automatically splits requests into batches containing at most 50,000 requests and 200 MB of JSONL input. It returns the batch IDs in submission order, and `fetch_and_process_batch()` merges their responses in the original head-proposition order. Head propositions with no candidate tails do not produce Batch API requests.

Pass the same `head_propositions` and `batch_tail_propositions` in the same order to both methods. Keep the model settings, prompt template, and `use_timestamp` unchanged between submission and fetching. `fetch_and_process_batch()` raises a `RuntimeError` if any OpenAI batch is incomplete or contains failed requests.

## Candidate Retrieval

Proposition Relation Extraction uses a Passage Retrieval component to find candidate tail propositions.

The head proposition itself is removed from its candidate tails.

The `prefilter_k` argument controls how many propositions are retrieved before filtering. The `top_k` argument controls how many candidates remain after filtering.

For multiple head propositions, `batch_retrieve_tail_propositions()` performs the same processing in batches.

## Timestamp Usage

If `use_timestamp` is `True`, each proposition must contain a `timestamp` in `YYYY-MM-DD` format. Candidate tails dated after the head proposition are excluded. The remaining candidates are ordered by timestamp before they are passed to the LLM. The timestamp is included in the prompt as `[As of YYYY-MM-DD]`.

If `use_timestamp` is `False`, no temporal filtering is applied to ensure that each tail proposition has the same timestamp as the head proposition or an earlier timestamp. Candidate tails may therefore be earlier than, simultaneous with, or later than the head proposition. Only the proposition text is included in the prompt.

## Custom Prompt Templates

A custom prompt template for `LLMPropositionRelationExtractor` supports the following placeholders.

| Placeholder | Required | Description |
|---|---|---|
| `{subject}` | Yes | Head proposition, optionally prefixed with its timestamp |
| `{object_list}` | Yes | Indexed list of candidate tail propositions, each optionally prefixed with its timestamp |

The component validates that both placeholders are present.

## Example

See [experiments/proposition_relation_extraction](../../experiments/proposition_relation_extraction) for runnable examples.
