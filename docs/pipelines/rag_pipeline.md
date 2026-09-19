# RAG (`kapipe.pipelines.RAGPipeline`)

**RAG** pipeline connects components for passage retrieval and question answering.

The pipeline does not define the behavior of the individual components. See the corresponding component documentation for their inputs, outputs, methods, and configuration.

## Component Flow

### Index construction:

```text
Passages (input)
↓ Passage Retrieval (indexing)
Retrieval index (output)
```

### Inference:

```text
Question (input)
↓ Passage Retrieval (search)
Retrieved passages (output)

Question and Retrieved passages (input)
↓ Question Answering
Answer (output)
```

## Components

| Constructor Argument | Component | Used During |
|---|---|---|
| `passage_retrieval` | [Passage Retrieval](../components/passage_retrieval.md) | Indexing and inference |
| `qa` | [Question Answering](../components/qa.md) | Inference |

All constructor arguments must be specified.

## Pipeline Methods

| Method | Description |
|---|---|
| `make_index()` | Builds a retrieval index over passages |
| `load_index()` | Loads the retrieval index for inference |
| `infer()` | Retrieves passages for each question and generates answers |

## Usage

### Initialize the Pipeline:

```python
from kapipe.pipelines import RAGPipeline

# Initialize the components
passage_retrieval = ...
qa = ...

# Instantiate the RAG pipeline
rag = RAGPipeline(
    passage_retrieval=passage_retrieval,
    qa=qa,
)
```

### Build the Index:

```python
# Build a retrieval index over passages
rag.make_index(
    passages=passages,
    index_dir="./indexes",
    passage_retrieval_indexing_kwargs={
        "batch_size": 1024,
    },
)
```

Values in `passage_retrieval_indexing_kwargs` are forwarded to the Passage Retrieval component. Omit arguments unsupported by the selected component.

### Load the Index:

```python
# Load the passage retrieval index
rag.load_index(
    index_dir="./indexes",
)
```

`load_index()` must be called before inference.

### Run Inference:

```python
# Retrieve passages and answer the questions
result_questions = rag.infer(
    questions=questions,
    top_k=5,
)
```

`top_k` controls the number of passages returned by Passage Retrieval.

### Use the OpenAI Batch API During Inference:

When the Question Answering component is `LLMQA` backed by `OpenAILLM`, inference can submit and fetch answer-generation requests through the OpenAI Batch API.
Passage Retrieval still runs during both calls.

```python
# Submit requests for inference
result_questions = rag.infer(
    questions=questions,
    top_k=5,
    batch_mode="submit",
    batch_dir="./batches",
)
assert result_questions is None

# Fetch the responses for inference
result_questions = rag.infer(
    questions=questions,
    top_k=5,
    batch_mode="fetch",
    batch_dir="./batches",
)
```

When `batch_mode` is specified, `batch_dir` is required.
Valid modes are `"submit"` and `"fetch"`.

## Indexing Outputs

`make_index()` creates `passage_retrieval_index/` under `index_dir`.

The contents of the retrieval index depend on the selected Passage Retrieval component.

## Inference Output

Each output preserves the fields returned by the Question Answering component and adds `contexts`, containing the passages returned by Passage Retrieval.

The pipeline does not save inference output automatically. The caller is responsible for saving `result_questions`.

## Example

See [experiments/rag_pipeline](../../experiments/rag_pipeline) for runnable examples.
