# Question Answering (`kapipe.qa`)

**Question Answering (QA)** answers a question with or without context passages.

This component takes a question and optional contexts, and returns the question with a generated answer.

## Input

A question is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `question_key` | `str` | Unique question identifier |
| `question` | `str` | Natural language question |
| `candidate_answers` | `list[str]` | Candidate answers, if available |

```json
{
    "question_key": "question#123",
    "question": "What does lithium carbonate induce?"
}
```

Contexts are optional.

Contexts are represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `question_key` | `str` | Unique question identifier |
| `contexts` | `list[dict]` | Context passages |

Each context passage conatins the following fields.

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Passage title, if available |
| `text` | `str` | Passage text |
| `score` | `float` | Retrieval score, if available |
| `rank` | `int` | Retrieval rank, if available |

```json
{
    "question_key": "question#123",
    "contexts": [
        {
            "title": "Lithium Carbonate and Related Health Conditions",
            "text": "This report examines the interconnections between Lithium Carbonate, ...",
            "score": 1.5991605520248413,
            "rank": 1
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
    "question": "What does lithium carbonate induce?",
    "output_answer": "Lithium Carbonate induces Depressive Disorder, Cyanosis, and Cardiac Arrhythmias.",
    "rationale": "The context passages indicate that Lithium Carbonate is associated with ...",
    "helpfulness_score": 0.9
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
model = ...

# Build an LLM-based QA component
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

# Build an LLM-based QA component
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
