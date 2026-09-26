# ProStruct-RAG (`kapipe.pipelines.ProStructRAGPipeline`)

**ProStruct-RAG** pipeline structures source passages as propositions and proposition relations for retrieval-augmented question answering.

Each indexing step can be run independently. Intermediate results are saved under `index_dir`, allowing later steps to resume from previously generated files.

The pipeline does not define the behavior of the individual components. See the corresponding component documentation for their inputs, outputs, methods, and configuration.

## Component Flow

### Index construction:

```text
Passages (input)
↓ Proposition Extraction
Propositions (output)

Propositions (input)
↓ Intermediate Passage Retrieval
↓ Proposition Relation Extraction
Triples (output)

Triples (input)
↓ Proposition Relation Refinement
Refined triples (output)

Refined triples (input)
↓ Passage Graph Construction
Graph (output)

Propositions (input)
↓ Passage Retrieval (indexing)
Retrieval index (output)
```

### Inference:

```text
Question (input)
↓ Passage Retrieval (search)
Anchor propositions (output)

Anchor propositions (input)
↓ Graph Retrieval
Neighborhood nodes and edges (output)

Neighborhood nodes and edges (input)
↓ Context Formatting
Structured textual context (output)

Question and Structured textual context (input)
↓ Question Answering
Answer (output)
```

## Components

| Constructor Argument | Component | Used During |
|---|---|---|
| `proposition_extraction` | [Proposition Extraction](../components/proposition_extraction.md) | Indexing |
| `intermediate_passage_retrieval` | [Passage Retrieval](../components/passage_retrieval.md) | Indexing |
| `proposition_relation_extraction` | [Proposition Relation Extraction](../components/proposition_relation_extraction.md) | Indexing |
| `proposition_relation_refinement` | [Proposition Relation Refinement](../components/proposition_relation_refinement.md) | Indexing |
| `passage_graph_construction` | [Passage Graph Construction](../components/passage_graph_construction.md) | Indexing |
| `passage_retrieval` | [Passage Retrieval](../components/passage_retrieval.md) | Indexing and inference |
| `graph_retrieval` | [Graph Retrieval](../components/graph_retrieval.md) | Inference |
| `context_formatting` | [Context Formatting](../components/context_formatting.md) | Inference |
| `qa` | [Question Answering](../components/qa.md) | Inference |

All constructor arguments must be specified.

## Pipeline Methods

| Method | Description |
|---|---|
| `make_index()` | Runs all indexing steps or one selected indexing step |
| `load_index()` | Loads the proposition retrieval index and proposition graph for inference |
| `infer()` | Retrieves and formats a proposition subgraph for each question and generates answers |

## Usage

### Initialize the Pipeline:

```python
from kapipe.pipelines import ProStructRAGPipeline

# Initialize the components
proposition_extraction = ...
intermediate_passage_retrieval = ...
proposition_relation_extraction = ...
proposition_relation_refinement = ...
passage_graph_construction = ...
passage_retrieval = ...
graph_retrieval = ...
context_formatting = ...
qa = ...

# Instantiate the ProStruct-RAG pipeline
prostruct_rag = ProStructRAGPipeline(
    proposition_extraction=proposition_extraction,
    intermediate_passage_retrieval=intermediate_passage_retrieval,
    proposition_relation_extraction=proposition_relation_extraction,
    proposition_relation_refinement=proposition_relation_refinement,
    passage_graph_construction=passage_graph_construction,
    passage_retrieval=passage_retrieval,
    graph_retrieval=graph_retrieval,
    context_formatting=context_formatting,
    qa=qa,
)
```

### Build the Index at Once:

```python
# Run all indexing steps
prostruct_rag.make_index(
    passages=passages,
    index_dir="./indexes",
    top_k=20,
    prefilter_k=100,
    search_batch_size=10,
    intermediate_passage_retrieval_indexing_kwargs={
        "batch_size": 1024,
    },
    passage_retrieval_indexing_kwargs={
        "batch_size": 1024,
    },
    use_timestamp_for_candidate_filtering_and_sorting=True,
)
```

`top_k`, `prefilter_k`, and `search_batch_size` control intermediate candidate retrieval before Proposition Relation Extraction.
Values in `intermediate_passage_retrieval_indexing_kwargs` and `passage_retrieval_indexing_kwargs` are forwarded to the corresponding Passage Retrieval components. Omit arguments unsupported by the selected components.
If `use_timestamp_for_candidate_filtering_and_sorting` is `True`, candidate tails later than the head proposition are removed and the selected tails are sorted chronologically. Every proposition must contain a `timestamp` in `YYYY-MM-DD` format. If it is `False`, candidates are not filtered by timestamp, and retrieval order is preserved.

### Build the Index Step by Step:

Set `target_component` to run only one indexing step.

```python
# Example 1: Run only Proposition Relation Extraction over the input under `index_dir`
prostruct_rag.make_index(
    passages=passages,
    index_dir="./indexes",
    top_k=20,
    prefilter_k=100,
    search_batch_size=10,
    use_timestamp_for_candidate_filtering_and_sorting=True,
    target_component="proposition_relation_extraction",
)

# Example 2: Run only Proposition Relation Refinement over the input under `index_dir`
prostruct_rag.make_index(
    passages=passages,
    index_dir="./indexes",
    top_k=20,
    prefilter_k=100,
    search_batch_size=10,
    use_timestamp_for_candidate_filtering_and_sorting=True,
    target_component="proposition_relation_refinement",
)
```

When run individually, steps after Proposition Extraction load their inputs from the standard filenames under `index_dir`.

### Use the OpenAI Batch API During Indexing:

The Proposition Extraction, Proposition Relation Extraction, and Proposition Relation Refinement steps support the OpenAI Batch API when their LLM component uses `OpenAILLM`.

```python
# Submit requests for Proposition Extraction
prostruct_rag.make_index(
    passages=passages,
    index_dir="./indexes",
    top_k=20,
    prefilter_k=100,
    search_batch_size=10,
    use_timestamp_for_candidate_filtering_and_sorting=True,
    target_component="proposition_extraction",
    batch_mode="submit",
    batch_dir="./batches",
)

# Fetch the responses for Proposition Extraction
prostruct_rag.make_index(
    passages=passages,
    index_dir="./indexes",
    top_k=20,
    prefilter_k=100,
    search_batch_size=10,
    use_timestamp_for_candidate_filtering_and_sorting=True,
    target_component="proposition_extraction",
    batch_mode="fetch",
    batch_dir="./batches",
)
```

When `batch_mode` is specified, `target_component` and `batch_dir` are required.
Valid modes are `"submit"` and `"fetch"`.
Running all indexing steps at once is not supported in batch mode.

### Load the Index:

```python
# Load the proposition retrieval index and directed graph
prostruct_rag.load_index(
    index_dir="./indexes",
)
```

`load_index()` must be called before inference.

### Run Inference:

```python
# Retrieve proposition subgraphs and answer the questions
result_questions = prostruct_rag.infer(
    questions=questions,
    top_k=10,
    hop_size=1,
    remove_same_timestamp_updates=True,
    append_question_timestamp=True,
)
```

`top_k` controls the number of anchor propositions returned by Passage Retrieval.
`hop_size` controls neighborhood expansion during Graph Retrieval.
If `remove_same_timestamp_updates` is `True`, `updates` edges whose head and tail timestamps are equal are excluded from Context Formatting.
If `append_question_timestamp` is `True`, the input question must contain a `timestamp`. The timestamp is appended to the question text as `(Date: <timestamp>)` before Question Answering.

### Use the OpenAI Batch API During Inference:

When the Question Answering component is `LLMQA` backed by `OpenAILLM`, inference can submit and fetch answer-generation requests through the OpenAI Batch API.
Passage Retrieval, Graph Retrieval, and Context Formatting still run during both calls.

```python
# Submit requests for inference
result_questions = prostruct_rag.infer(
    questions=questions,
    top_k=10,
    hop_size=1,
    batch_mode="submit",
    batch_dir="./batches",
)
assert result_questions is None

# Fetch the responses for inference
result_questions = prostruct_rag.infer(
    questions=questions,
    top_k=10,
    hop_size=1,
    batch_mode="fetch",
    batch_dir="./batches",
)
```

When `batch_mode` is specified, `batch_dir` is required.
Valid modes are `"submit"` and `"fetch"`.

## Indexing Outputs

`make_index()` creates the following artifacts under `index_dir`.

| Path | Created By |
|---|---|
| `propositions.jsonl` | Proposition Extraction |
| `intermediate_passage_retrieval_index/` | Intermediate Passage Retrieval |
| `triples.json` | Proposition Relation Extraction |
| `refined_triples.json` | Proposition Relation Refinement |
| `graph.graphml` | Passage Graph Construction |
| `passage_retrieval_index/` | Passage Retrieval indexing |

The contents of both retrieval index directories depend on the selected Passage Retrieval components.

## Inference Output

The output preserves the fields returned by the Question Answering component and adds the following intermediate results.

| Field | Description |
|---|---|
| `anchor_contexts` | Propositions returned by Passage Retrieval |
| `graph_contexts` | Nodes and edges returned by Graph Retrieval |
| `formatted_contexts` | Graph content converted into a textual QA context with passage key `<question_key>/context#0000` |

The pipeline does not save inference output automatically. The caller is responsible for saving `result_questions`.

## Example

See [experiments/prostruct_rag_pipeline_emnlp2026](../../experiments/prostruct_rag_pipeline_emnlp2026) for runnable examples.
