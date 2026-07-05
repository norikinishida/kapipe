# Question Answering (`kapipe.qa`)

**Question Answering (QA)** answers a question with or without context passages.

This component takes a question and optional contexts, and returns the question with a generated answer.

## Input

A question is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `question_key` | `str` | Unique question identifier |
| `question` | `str` | Natural language question |

```json
{
    "question_key": "question#123",
    "question": "Which interventions attenuated, prevented, antagonized, or reduced opioid-induced muscle rigidity in rats?"
}
```

Contexts are optional.

Contexts are represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `question_key` | `str` | Unique question identifier |
| `contexts` | `list[dict]` | Context passages |

Each context passage contains the following fields.

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Passage title, if available |
| `text` | `str` | Passage text |

```json
{
    "question_key": "question#123",
    "contexts": [
        {
            "title": "Ketanserin pretreatment reverses alfentanil-induced muscle rigidity.",
            "text": "Systemic pretreatment with ketanserin, a relatively specific type-2 serotonin receptor antagonist, significantly attenuated ..."
        },
        {
            "title": "Involvement of locus coeruleus and noradrenergic neurotransmission in fentanyl-induced muscular rigidity in the rat.",
            "text": "Whereas muscular rigidity is a well-known side effect that is associated with high-dose fentanyl anesthesia, a paucity of ..."
        },
        ...
    ]
}
```

## Output

The output preserves the input question fields and adds answer-related fields.

| Field | Type | Description |
|---|---|---|
| `output_answer` | `str` | Generated answer |
| `rationale` | `str` | Generated rationale, if parsed |
| `helpfulness_score` | `float` | Parsed helpfulness score, if available |

```json
{
    "question_key": "question#123",
    "question": "Which interventions attenuated, prevented, antagonized, or reduced opioid-induced muscle rigidity in rats?",
    "output_answer": "Ketanserin, electrolytic lesions of the locus coeruleus, and prazosin attenuated, prevented, antagonized, or reduced opioid-induced muscle rigidity in rats.",
    "rationale": "The context passages provide information on different interventions that affect opioid-induced muscle rigidity in rats. Passage [1] mentions that ...",
    "helpfulness_score": 1.0
}
```

## Supported Methods

| Method | Description |
|---|---|
| LLM-based QA | Answers questions using a proprietary or open-source LLM |

## Usage

### LLM-based QA with Contexts:

```python
from kapipe.qa import LLMQA

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

# Instantiate the LLM-based QA component
answerer = LLMQA(
    model=model,
    prompt_template_name_or_path="qa_03_with_context",
    n_contexts=10,
)

# Answer a question with retrieved contexts
result_question = answerer.answer(
    question=question,
    contexts_for_question=contexts_for_question,
)
```

### LLM-based QA without Contexts:

```python
from kapipe.qa import LLMQA

# Instantiate your LLM wrapper
model = ...

# Instantiate the LLM-based QA component
answerer = LLMQA(
    model=model,
    prompt_template_name_or_path="qa_03_without_context",
)

# Answer a question without retrieved contexts
result_question = answerer.answer(
    question=question,
    contexts_for_question=None,
)
```

## Context Usage

If `contexts_for_question` is provided, the retrieved passages are inserted into the QA prompt.

The `n_contexts` argument controls how many passages are used. If `n_contexts` is `-1`, all contexts are used.

## Example

See [experiments/qa](../../experiments/qa) for runnable examples.
