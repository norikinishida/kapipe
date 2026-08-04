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
    "question_key": "question#001",
    "question": "Which metastatic breast cancer treatments were linked to an adverse event also reported in dogs treated with lomustine?"
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
    "question_key": "question#001",
    "contexts": [
        {
            "title": "CCNU (lomustine) toxicity in dogs: a retrospective study (2002-07).",
            "text": "OBJECTIVE: To describe the incidence of haematological, renal, hepatic and gastrointestinal toxicities in tumour-bearing dogs ...",
            "score": 0.7051769495010376,
            "rank": 1
        },
        {
            "title": "Reduced cardiotoxicity and preserved antitumor efficacy of liposome-encapsulated doxorubicin and cyclophosphamide compared with ...",
            "text": "PURPOSE: To determine whether Myocet (liposome-encapsulated doxorubicin; The Liposome Company, Elan Corporation, Princeton, ...",
            "score": 0.5899654626846313,
            "rank": 2
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
    "question_key": "question#001",
    "question": "Which metastatic breast cancer treatments were linked to an adverse event also reported in dogs treated with lomustine?",
    "output_answer": "Conventional doxorubicin and cyclophosphamide (AC) treatment for metastatic breast cancer was linked to neutropenia, ...",
    "rationale": "The context passages describe various treatments and their associated adverse events. In passage [1], dogs treated with ...",
    "helpfulness_score": 0.95
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
    max_new_tokens=8192,
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

## Custom Prompt Templates

A custom prompt template for `LLMQA` supports the following placeholders.

| Placeholder | Required | Description |
|---|---|---|
| `{contexts_prompt}` | No | Context block provided for the question |
| `{test_case_prompt}` | Yes | Current question and its candidate answer options (if available) |

`{contexts_prompt}` is used by templates that answer questions with contexts. It can be omitted from templates that answer questions without contexts.

The component validates that `{test_case_prompt}` is present.

## Example

See [experiments/qa](../../experiments/qa) for runnable examples.
