# RAG (`kapipe.pipelines.RAGPipeline`)

**Retrieval-Augmented Generation (RAG)** pipeline connects Passage Retrieval and Question Answering components.

The pipeline delegates retrieval and answer generation to the supplied components. See the component documentation for their inputs, outputs, methods, and configuration.

## Component Flow

Index construction:

```text
Passages (input)
  → Passage Retrieval
    → Retrieval index (output)
```

Inference:

```text
Question (input)
  → Passage Retrieval
  → Question Answering
    → Answer (output)
```

The retrieved passages are preserved in the pipeline output as `contexts`.

## Components

| Constructor Argument | Component | Required |
|---|---|---|
| `passage_retrieval` | [Passage Retrieval](../components/passage_retrieval.md) | Yes |
| `qa` | [Question Answering](../components/qa.md) | Yes |

The components must be instantiated before they are passed to the pipeline.

## Pipeline Methods

| Method | Description |
|---|---|
| `make_index()` | Builds a retrieval index over passages |
| `load_index()` | Loads an existing retrieval index |
| `infer()` | Retrieves passages for one question and generates an answer |

### `make_index()`

```python
rag.make_index(
    passages=passages,
    index_dir="./indexes",
    batch_size=64,
)
```

Additional keyword arguments, such as `batch_size`, are passed directly to the Passage Retrieval component.

### `load_index()`

```python
rag.load_index(
    index_dir="./indexes",
)
```

### `infer()`

```python
result_question = rag.infer(
    question=question,
    top_k=5,
)
```

`top_k` is passed to the Passage Retrieval component.

## Usage

```python
from kapipe.pipelines import RAGPipeline


# Initialize the components before constructing the pipeline
passage_retrieval = ...
qa = ...

# Connect Passage Retrieval and Question Answering
rag = RAGPipeline(
    passage_retrieval=passage_retrieval,
    qa=qa,
)

# Build a retrieval index over passages
rag.make_index(
    passages=passages,
    index_dir="./indexes",
    batch_size=64,
)

# Retrieve passages and answer one question
result_question = rag.infer(
    question=question,
    top_k=5,
)
```

To reuse an existing index:

```python
from kapipe.pipelines import RAGPipeline


# Initialize components compatible with the existing index
passage_retrieval = ...
qa = ...

# Connect Passage Retrieval and Question Answering
rag = RAGPipeline(
    passage_retrieval=passage_retrieval,
    qa=qa,
)

# Load the existing retrieval index
rag.load_index(
    index_dir="./indexes",
)

# Retrieve passages and answer one question
result_question = rag.infer(
    question=question,
    top_k=5,
)
```

## Intermediate Files

The Passage Retrieval component saves its index under `index_dir`.

The index format and filenames depend on the selected Passage Retrieval component. See [Passage Retrieval](../components/passage_retrieval.md) for component-specific behavior.

`RAGPipeline` does not save inference results automatically. The caller is responsible for saving the returned questions.

## Example

See [experiments/rag_pipeline](../../experiments/rag_pipeline) for a runnable example.