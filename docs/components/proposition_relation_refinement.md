# Proposition Relation Refinement (`kapipe.proposition_relation_refinement`)

**Proposition Relation Refinement** verifies and corrects extracted relations between propositions.

This component takes a proposition relation record and uses an LLM to refine its relation label and explanation.

## Input

The input is a proposition relation record.

Each relation record contains the following fields.

| Field | Type | Description |
|---|---|---|
| `head` | `dict` | Head proposition |
| `relation` | `str` | Predicted relation from the head proposition to the tail proposition |
| `tail` | `dict` | Tail proposition |
| `explanation` | `str` | Predicted explanation of the relation |

Each proposition must contain `passage_key` and `text`. A `timestamp` field in `YYYY-MM-DD` format is also required if temporal information is used.

```json
{
    "head": {
        "passage_key": "proposition#001",
        "text": "An independent audit found that the Northbridge payment system processed 99.9% of transactions within two seconds in February 2025.",
        "timestamp": "2025-03-01"
    },
    "relation": "supports",
    "tail": {
        "passage_key": "proposition#002",
        "text": "An independent audit found that the Northbridge payment system processed only 82.0% of transactions within two seconds in February 2025.",
        "timestamp": "2025-03-01"
    },
    "explanation": "Both audit results describe the processing performance in February 2025."
}
```

## Output

The output is a refined proposition relation record.

The head and tail propositions are preserved. The refined relation and explanation replace the original prediction, which is retained in the `pre_refinement_relation` and `pre_refinement_explanation` fields.

| Field | Type | Description |
|---|---|---|
| `head` | `dict` | Head proposition |
| `relation` | `str` | Refined relation from the head proposition to the tail proposition |
| `tail` | `dict` | Tail proposition |
| `explanation` | `str` | Refined explanation of the relation |
| `pre_refinement_relation` | `str` | Relation before refinement |
| `pre_refinement_explanation` | `str` | Explanation before refinement |

```json
{
    "head": {
        "passage_key": "proposition#001",
        "text": "An independent audit found that the Northbridge payment system processed 99.9% of transactions within two seconds in February 2025.",
        "timestamp": "2025-03-01"
    },
    "relation": "contradicts",
    "tail": {
        "passage_key": "proposition#002",
        "text": "An independent audit found that the Northbridge payment system processed only 82.0% of transactions within two seconds in February 2025.",
        "timestamp": "2025-03-01"
    },
    "explanation": "The reported processing rates of 99.9% and 82.0% for February 2025 are incompatible.",
    "pre_refinement_relation": "supports",
    "pre_refinement_explanation": "Both audit results describe the processing performance in February 2025."
}
```

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
| LLM-based Proposition Relation Refinement | Verifies and corrects proposition relation labels and explanations using a proprietary or open-source LLM |

## Usage

### LLM-based Proposition Relation Refinement:

```python
from kapipe.proposition_relation_refinement import LLMPropositionRelationRefiner

# Instantiate your LLM wrapper
# OpenAI LLM
from kapipe.llms import OpenAILLM
model = OpenAILLM(
    model_name="gpt-5.4",
    max_new_tokens=2048,
)
# HuggingFace LLM
from kapipe.llms import HuggingFaceLLM
model = HuggingFaceLLM(
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    max_new_tokens=2048,
    quantization_bits=4,
)

# Instantiate the LLM-based Proposition Relation Refinement component
refiner = LLMPropositionRelationRefiner(
    model=model,
    prompt_template_name_or_path=(
        "proposition_relation_refinement_01_with_timestamp"
    ),
    use_timestamp=True,
)

# Refine a proposition relation record
refined_triple = refiner.refine(triple=triple)
```

## Handling `NOREL`

The `refine()` method returns a relation record even when the refined relation is `NOREL`.

Remove records classified as `NOREL` after refinement if only valid relations should be retained.

```python
# Refine proposition relations and retain valid relations
refined_triples = []
for triple in triples:
    refined_triple = refiner.refine(triple=triple)
    if refined_triple["relation"] == "NOREL":
        continue
    refined_triples.append(refined_triple)
```

## Timestamp Usage

If `use_timestamp` is `True`, both propositions must contain a `timestamp` in `YYYY-MM-DD` format. The component does not validate or change the temporal order of the head and tail propositions. Each timestamp is included in the prompt as `[As of YYYY-MM-DD]`.

If `use_timestamp` is `False`, no `timestamp` field is required, and only the proposition text is included in the prompt.

## Custom Prompt Templates

A custom prompt template for `LLMPropositionRelationRefiner` supports the following placeholders.

| Placeholder | Required | Description |
|---|---|---|
| `{subject}` | Yes | Head proposition, optionally prefixed with its timestamp |
| `{object}` | Yes | Tail proposition, optionally prefixed with its timestamp |
| `{predicted_relation}` | Yes | Relation before refinement |
| `{predicted_explanation}` | Yes | Explanation before refinement |

The component validates that all four placeholders are present.

## Example

See [experiments/proposition_relation_refinement](../../experiments/proposition_relation_refinement) for runnable examples.
