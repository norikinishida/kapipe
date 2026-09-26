# Proposition Relation Extraction (`kapipe.proposition_relation_extraction`)

**Proposition Relation Extraction** identifies directed relations between propositions.

This component classifies relations from one head proposition to given tail propositions.

## Input

The input is one head proposition and a list of tail propositions.

Each proposition is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `passage_key` | `str` | Unique proposition identifier |
| `text` | `str` | Proposition text |
| `timestamp` | `str` | Date in `YYYY-MM-DD` format, if temporal relations are used |

Additional metadata fields are preserved during relation extraction.

```python
head_proposition = {
        "passage_key": "proposition#001",
        "text": "An independent audit found that the Northbridge payment system processed 99.9% of transactions within two seconds in February 2025."
}

tail_propositions = [
    {
        "passage_key": "proposition#002",
        "text": "A preliminary report found that the Northbridge payment system processed 97.0% of transactions within two seconds in January 2025."
    },
    ...
]
```

## Output

The output is a list of directed relation records.

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
            "text": "An independent audit found that the Northbridge payment system processed 99.9% of transactions within two seconds in February 2025."
        },
        "relation": "updates",
        "tail": {
            "passage_key": "proposition#002",
            "text": "A preliminary report found that the Northbridge payment system processed 97.0% of transactions within two seconds in January 2025."
        },
        "explanation": "The February audit updates the January processing rate from 97.0% to 99.9%."
    },
    ...
]
```

Entries classified as `NOREL` are excluded from the output.

## Relation Labels

The component does not define a fixed relation-label inventory. The selected prompt template determines the relation scheme.

The default `proposition_relation_extraction_01` prompt predicts a concise relation label for each proposition pair. It predicts `NOREL` when no direct relation applies.

The parser accepts any string as a relation label. It does not restrict predictions to labels demonstrated in the prompt examples.

## Supported Methods

| Method | Description |
|---|---|
| LLM-based Proposition Relation Extraction | Classifies given proposition pairs using a proprietary or open-source LLM |

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

# Instantiate the LLM-based Proposition Relation Extraction component
extractor = LLMPropositionRelationExtractor(
    model=model,
    prompt_template_name_or_path="proposition_relation_extraction_01",
    use_timestamp=False,
)

# Extract directed relations from the head proposition to the given tails
triples = extractor.extract(
    head_proposition=head_proposition,
    tail_propositions=tail_propositions,
)
```

## OpenAI Batch API

`LLMPropositionRelationExtractor` supports the OpenAI Batch API when its model is an `OpenAILLM` instance.

```python
# Submit all relation-extraction prompts to one or more OpenAI batches
batch_ids = extractor.submit_batch(
    head_propositions=[head_proposition],
    batch_tail_propositions=[tail_propositions],
)

# Fetch and process the results after all OpenAI batches are complete
triples = extractor.fetch_and_process_batch(
    head_propositions=[head_proposition],
    batch_tail_propositions=[tail_propositions],
    batch_ids=batch_ids,
)
```

`submit_batch()` automatically splits requests into batches containing at most 50,000 requests and 200 MB of JSONL input. It returns the batch IDs in submission order, and `fetch_and_process_batch()` merges their responses in the original head-proposition order. Head propositions with no candidate tails do not produce Batch API requests.

Pass the same `head_propositions` and `batch_tail_propositions` in the same order to both methods. Keep the model settings, prompt template, and `use_timestamp` unchanged between submission and fetching. `fetch_and_process_batch()` raises a `RuntimeError` if any OpenAI batch is incomplete or contains failed requests.

## Timestamp Usage

If `use_timestamp` is `True`, each proposition must contain a `timestamp` in `YYYY-MM-DD` format. During relation classification, the timestamp is included in the prompt as `[As of YYYY-MM-DD]`. This is useful when the prompt instructs the model to classify relations involving temporal information.

If temporal information is not relevant to the classification, `use_timestamp=False` may be sufficient. In this case, no `timestamp` field is required, and only the proposition text is included in the prompt.

## Custom Prompt Templates

`prompt_template_name_or_path` accepts either the name of a built-in prompt template ([`kapipe/proposition_relation_extraction/prompt_templates/*.txt`](../../kapipe/proposition_relation_extraction/prompt_templates)) or a path to a user-defined prompt template.

A custom prompt template for `LLMPropositionRelationExtractor` supports the following placeholders.

| Placeholder | Required | Description |
|---|---|---|
| `{subject}` | Yes | Head proposition, optionally prefixed with its timestamp |
| `{object_list}` | Yes | Indexed list of candidate tail propositions, each optionally prefixed with its timestamp |

The component validates that both placeholders are present.

## Example

See [experiments/proposition_relation_extraction](../../experiments/proposition_relation_extraction) for runnable examples.
