# ProStruct-RAG (`kapipe.pipelines.ProStructRAGPipeline`)

**ProStruct-RAG** pipeline structures source passages as propositions and proposition relations for retrieval-augmented question answering.

Each indexing step can be run independently. Intermediate results are saved under `index_dir`, allowing later steps to resume from previously generated files.

The pipeline does not define the behavior of the individual components. See the corresponding component documentation for their inputs, outputs, methods, and configuration.

## Component Flow

### Index construction:

```text
Passages (input)
→ Proposition Extraction
  → Propositions (output)

Propositions (input)
→ Proposition Relation Extraction
  → Triples (output)

Triples (input)
→ Proposition Relation Refinement
  → Refined triples (output)

Refined triples (input)
→ Passage Graph Construction
  → Graph (output)

Propositions (input)
→ Passage Retrieval indexing
  → Retrieval index (output)
```

### Inference:

```text
Question (input)
→ Passage Retrieval search
  → Anchor propositions (output)

Anchor propositions (input)
→ Graph Retrieval
  → Neighborhood nodes and edges (output)

Neighborhood nodes and edges (input)
→ Context Formatting
  → Structured textual context (output)

Question and Structured textual context (input)
→ Question Answering
  → Answer (output)
```

## Components

| Constructor Argument | Component | Used During |
|---|---|---|
| `proposition_extraction` | [Proposition Extraction](../components/proposition_extraction.md) | Indexing |
| `proposition_relation_extraction` | [Proposition Relation Extraction](../components/proposition_relation_extraction.md) | Indexing |
| `proposition_relation_refinement` | [Proposition Relation Refinement](../components/proposition_relation_refinement.md) | Indexing |
| `passage_graph_construction` | [Passage Graph Construction](../components/passage_graph_construction.md) | Indexing |
| `passage_retrieval` | [Passage Retrieval](../components/passage_retrieval.md) | Indexing and inference |
| `graph_retrieval` | [Graph Retrieval](../components/graph_retrieval.md) | Inference |
| `context_formatting` | [Context Formatting](../components/context_formatting.md) | Inference |
| `qa` | [Question Answering](../components/qa.md) | Inference |

All constructor arguments must be specified. Components unused by the intended operation may be set to `None`.

Calling a method without a component required by that operation raises an error.

## Pipeline Methods

| Method | Description |
|---|---|
| `make_index()` | Runs all indexing steps or one selected indexing step |
| `load_index()` | Loads the proposition retrieval index and proposition graph for inference |
| `infer()` | Retrieves and formats a proposition subgraph for one question and generates an answer |

## Usage

### Initialize the Pipeline:

```python
from kapipe.pipelines import ProStructRAGPipeline


# Initialize the components before constructing the pipeline
proposition_extraction = ...
proposition_relation_extraction = ...
proposition_relation_refinement = ...
passage_graph_construction = ...
passage_retrieval = ...
graph_retrieval = ...
context_formatting = ...
qa = ...

# Connect the initialized components
prostruct_rag = ProStructRAGPipeline(
    proposition_extraction=proposition_extraction,
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
    node_id_key="proposition_id",
    source_id_key="article_id",
    proposition_relation_extraction_indexing_kwargs={
        "batch_size": 1024,
    },
    passage_retrieval_indexing_kwargs={
        "batch_size": 1024,
    },
)
```

`top_k`, `prefilter_k`, and `search_batch_size` control candidate retrieval during Proposition Relation Extraction.

Values in `proposition_relation_extraction_indexing_kwargs` and `passage_retrieval_indexing_kwargs` are forwarded to the corresponding retrieval components. Omit arguments unsupported by the selected components.

### Build the Index Step by Step:

Set `target_component` to run only one indexing step.

| `target_component` | Input |
|---|---|
| `proposition_extraction` | `passages` |
| `proposition_relation_extraction` | Propositions |
| `proposition_relation_refinement` | Extracted triples |
| `passage_graph_construction` | Propositions and refined triples |
| `passage_retrieval_indexing` | Propositions |

```python
# Example 1: Run only Proposition Relation Extraction over the specified input
prostruct_rag.make_index(
    passages=None,
    index_dir="./indexes",
    top_k=20,
    prefilter_k=100,
    search_batch_size=10,
    node_id_key="proposition_id",
    source_id_key="article_id",
    target_component="proposition_relation_extraction",
    input_artifact_paths={
        "propositions": "/path/to/propositions.json",
    },
)

# Example 2: Run only Proposition Relation Refinement over the input under `index_dir`
prostruct_rag.make_index(
    passages=None,
    index_dir="./indexes",
    top_k=20,
    prefilter_k=100,
    search_batch_size=10,
    node_id_key="proposition_id",
    source_id_key="article_id",
    target_component="proposition_relation_refinement",
    input_artifact_paths=None,
)
```

The component required by the selected step must be initialized. Other constructor arguments can be `None`.

If `input_artifact_paths` is omitted, the selected step loads its inputs from the standard filenames under `index_dir`.

The supported artifact overrides are:

| Key | Artifact |
|---|---|
| `propositions` | Proposition records in JSONL format |
| `triples` | Extracted proposition relation records in JSON format |
| `refined_triples` | Refined proposition relation records in JSON format |

### Load the Index:

```python
# Load the proposition retrieval index and directed graph
prostruct_rag.load_index(
    index_dir="./indexes",
)
```

`load_index()` must be called before inference, including when the same pipeline instance created the index. Graph Retrieval is initialized from the saved graph during this call.

### Run Inference:

```python
# Retrieve a proposition subgraph and answer one question
result_question = prostruct_rag.infer(
    question=question,
    top_k=10,
    hop_size=1,
    node_id_key="proposition_id",
    remove_same_timestamp_updates=True,
    append_question_timestamp=True,
)
```

`top_k` controls the number of anchor propositions returned by Passage Retrieval. `hop_size` controls neighborhood expansion during Graph Retrieval.

The `node_id_key` used during inference must match the key used during Passage Graph Construction.

## Indexing Outputs

`make_index()` creates the following artifacts under `index_dir`.

| Path | Created By |
|---|---|
| `propositions.jsonl` | Proposition Extraction |
| `intermediate_passage_retrieval_index/` | Candidate retrieval for Proposition Relation Extraction |
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
| `formatted_contexts` | Graph content converted into textual QA context |

If `append_question_timestamp` is `True`, the input question must contain a `timestamp`. The timestamp is appended to the question text as `(Date: <timestamp>)` before Question Answering.

If `remove_same_timestamp_updates` is `True`, `updates` edges whose head and tail timestamps are equal are excluded from Context Formatting. They remain present in `graph_contexts`.

The pipeline does not save inference output automatically. The caller is responsible for saving `result_question`.

## Proposition Node Identifiers

During Proposition Extraction, the pipeline adds a node identifier to each proposition that does not already contain `node_id_key`.

The generated identifier has the following form.

```text
<source_id>/proposition<zero-padded proposition index>
```

For example, `node_id_key="proposition_id"` and `source_id_key="article_id"` can produce:

```text
article#001/proposition0000
```

The proposition index starts from zero for each source passage. Each source passage must contain `source_id_key` when an identifier must be generated.



## Example

See [experiments/prostruct_rag_pipeline_emnlp2026](../../experiments/prostruct_rag_pipeline_emnlp2026) for runnable examples.
