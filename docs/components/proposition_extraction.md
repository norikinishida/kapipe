# Proposition Extraction (`kapipe.proposition_extraction`)

**Proposition Extraction** decomposes a passage into simple, self-contained factual statements.

This component takes a passage and returns a list of propositions that can be used for retrieval or graph construction.

## Input

A *passage* is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `passage_key` | `str` | Unique passage identifier |
| `title` | `str` | Title, if available |
| `text` | `str` | Body text |

Additional metadata fields are preserved in the output.

```json
{
    "passage_key": "passage#001",
    "title": "Effect of calcium chloride and 4-aminopyridine therapy on desipramine toxicity in rats.",
    "text": "BACKGROUND: Hypotension is a major contributor to mortality in tricyclic antidepressant overdose. Recent data suggest that tricyclic antidepressants inhibit calcium influx in some tissues. ...",
    "source": "...",
    "timestamp": "..."
}
```

## Output

The output is a list of propositions.

Each proposition contains one extracted factual statement and preserves the input metadata other than `passage_key`, `title`, `text`, and `source_passage_key`.

| Field | Type | Description |
|---|---|---|
| `passage_key` | `str` | Proposition identifier derived from the source passage key |
| `text` | `str` | Extracted factual statement |
| `source_passage_key` | `str` | Passage key of the source passage |

```json
[
    {
        "passage_key": "passage#001/proposition#0000",
        "text": "Effect of calcium chloride and 4-aminopyridine therapy on desipramine toxicity in rats.",
        "source_passage_key": "passage#001",
        "source": "...",
        "timestamp": "..."
    },
    {
        "passage_key": "passage#001/proposition#0001",
        "text": "Hypotension is a major contributor to mortality in tricyclic antidepressant overdose.",
        "source_passage_key": "passage#001",
        "source": "...",
        "timestamp": "..."
    },
    {
        "passage_key": "passage#001/proposition#0002",
        "text": "Recent data suggest that tricyclic antidepressants inhibit calcium influx in some tissues.",
        "source_passage_key": "passage#001",
        "source": "...",
        "timestamp": "..."
    },
    ...
]
```

If the input passage contains a non-empty `title` field, its stripped value is inserted as the first proposition.

## Supported Methods

| Method | Description |
|---|---|
| LLM-based Proposition Extraction | Extracts propositions using a proprietary or open-source LLM |

## Usage

### LLM-based Proposition Extraction:

```python
from kapipe.proposition_extraction import LLMPropositionExtractor

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

# Instantiate the LLM-based Proposition Extraction component
extractor = LLMPropositionExtractor(
    model=model,
    prompt_template_name_or_path="proposition_extraction_01",
)

# Extract propositions from a passage
propositions = extractor.extract(passage=passage)
```

## OpenAI Batch API

`LLMPropositionExtractor` supports the OpenAI Batch API when its model is an `OpenAILLM` instance.

```python
# Submit all passage prompts to one or more OpenAI batches
batch_ids: list[str] = extractor.submit_batch(passages=passages)

# Fetch and process the results after all OpenAI batches are complete
propositions = extractor.fetch_and_process_batch(
    passages=passages,
    batch_ids=batch_ids,
)
```

`submit_batch()` automatically splits requests into batches containing at most 50,000 requests and 200 MB of JSONL input. It returns the batch IDs in submission order, and `fetch_and_process_batch()` merges their responses in the original passage order.

Pass the same `passages` in the same order to both methods. Keep the model settings and prompt template unchanged between submission and fetching. `fetch_and_process_batch()` raises a `RuntimeError` if any OpenAI batch is incomplete or contains failed requests.

## Metadata Preservation

Proposition Extraction preserves metadata fields other than `passage_key`, `title`, `text`, and `source_passage_key`.

Each extracted proposition receives the same metadata as the input passage.

Each proposition receives a `passage_key` of the form `<source_passage_key>/proposition#<zero-padded proposition index>`. The input `passage_key` is stored as `source_passage_key`.

## Custom Prompt Templates

A custom prompt template for `LLMPropositionExtractor` supports the following placeholder.

| Placeholder | Required | Description |
|---|---|---|
| `{input_text}` | Yes | Passage text from which propositions are extracted |

The passage text is inserted into `{input_text}` before the prompt is passed to the LLM.

The component validates that `{input_text}` is present.

## Example

See [experiments/proposition_extraction](../../experiments/proposition_extraction) for runnable examples.
