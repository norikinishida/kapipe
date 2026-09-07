# Proposition Extraction (`kapipe.proposition_extraction`)

**Proposition Extraction** decomposes a passage into simple, self-contained factual statements.

This component takes a passage and returns a list of propositions that can be used for retrieval or graph construction.

## Input

A *passage* is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Title, if available |
| `text` | `str` | Body text |

Additional metadata fields are preserved in the output.

```json
{
    "title": "Effect of calcium chloride and 4-aminopyridine therapy on desipramine toxicity in rats.",
    "text": "BACKGROUND: Hypotension is a major contributor to mortality in tricyclic antidepressant overdose. Recent data suggest that tricyclic antidepressants inhibit calcium influx in some tissues. ...",
    "source": "...",
    "timestamp": "..."
}
```

## Output

The output is a list of propositions.

Each proposition contains one extracted factual statement and preserves the input metadata other than `title` and `text`.

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Extracted factual statement |

```json
[
    {
        "text": "Effect of calcium chloride and 4-aminopyridine therapy on desipramine toxicity in rats.",
        "source": "...",
        "timestamp": "...",
    },
    {
        "text": "Hypotension is a major contributor to mortality in tricyclic antidepressant overdose.",
        "source": "...",
        "timestamp": "..."
    },
    {
        "text": "Recent data suggest that tricyclic antidepressants inhibit calcium influx in some tissues.",
        "source": "...",
        "timestamp": "..."
    },
    ...
]
```

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
    include_title_as_proposition=True,
)

# Extract propositions from a passage
propositions = extractor.extract(passage=passage)
```

## Title Inclusion

If `include_title_as_proposition` is `True`, a non-empty passage title is inserted as the first proposition.

If the title is missing or empty, it is not added.

## Metadata Preservation

Proposition Extraction preserves metadata fields other than `title` and `text`.

Each extracted proposition receives the same metadata as the input passage.

## Custom Prompt Templates

A custom prompt template for `LLMPropositionExtractor` supports the following placeholder.

| Placeholder | Required | Description |
|---|---|---|
| `{input_text}` | Yes | Passage text from which propositions are extracted |

The passage text is inserted into `{input_text}` before the prompt is passed to the LLM.

The component validates that `{input_text}` is present.

## Example

See [experiments/proposition_extraction](../../experiments/proposition_extraction) for runnable examples.
