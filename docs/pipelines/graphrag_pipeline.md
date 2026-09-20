# GraphRAG (`kapipe.pipelines.GraphRAGPipeline`)

**GraphRAG** pipeline connects components for triple extraction, knowledge graph organization, retrieval, and question answering.

Each indexing step can be run independently. Intermediate results are saved under `index_dir`, allowing later steps to resume from previously generated files.

The pipeline does not define the behavior of the individual components. See the corresponding component documentation for their inputs, outputs, methods, and configuration.

## Component Flow

### Index construction:

```text
Documents (input)
↓ Named Entity Recognition
Documents with mentions (output)

Documents with mentions (input)
↓ Entity Disambiguation (Retrieval)
Documents with candidate entities (output)

Documents with candidate entities (input)
↓ Entity Disambiguation (Reranking)
Documents with entities (output)

Documents with entities (input)
↓ Document-level Relation Extraction
Documents with triples (output)

Documents with triples (input)
↓ Entity Graph Construction
Entity graph (output)

Entity graph (input)
↓ Community Clustering
Communities (output)

Entity graph and Communities (input)
↓ Report Generation
Community reports (output)

Community reports (input)
↓ Chunking
Chunked reports (output)

Chunked reports (input)
↓ Passage Retrieval (indexing)
Retrieval index (output)
```

### Inference:

```text
Question (input)
↓ Passage Retrieval (search)
Retrieved chunks (output)

Question and Retrieved chunks (input)
↓ Question Answering
Answer (output)
```

## Components

| Constructor Argument | Component | Used During |
|---|---|---|
| `ner` | [Named Entity Recognition](../components/ner.md) | Indexing |
| `ed_retrieval` | [Entity Disambiguation (Retrieval)](../components/ed_retrieval.md) | Indexing |
| `ed_reranking` | [Entity Disambiguation (Reranking)](../components/ed_reranking.md) | Indexing |
| `docre` | [Document-level Relation Extraction](../components/docre.md) | Indexing |
| `entity_graph_construction` | [Entity Graph Construction](../components/entity_graph_construction.md) | Indexing |
| `community_clustering` | [Community Clustering](../components/community_clustering.md) | Indexing |
| `report_generation` | [Report Generation](../components/report_generation.md) | Indexing |
| `chunker` | [Chunking](../components/chunking.md) | Indexing |
| `passage_retrieval` | [Passage Retrieval](../components/passage_retrieval.md) | Indexing and inference |
| `qa` | [Question Answering](../components/qa.md) | Inference |

All constructor arguments must be specified.

## Pipeline Methods

| Method | Description |
|---|---|
| `make_index()` | Runs all indexing steps or one selected indexing step |
| `load_index()` | Loads the retrieval index for inference |
| `infer()` | Retrieves chunks for each question and generates answers |

## Usage

### Initialize the Pipeline:

```python
from kapipe.pipelines import GraphRAGPipeline

# Initialize the components
ner = ...
ed_retrieval = ...
ed_reranking = ...
docre = ...
entity_graph_construction = ...
community_clustering = ...
report_generation = ...
chunker = ...
passage_retrieval = ...
qa = ...

# Instantiate the GraphRAG pipeline
graphrag = GraphRAGPipeline(
    ner=ner,
    ed_retrieval=ed_retrieval,
    ed_reranking=ed_reranking,
    docre=docre,
    entity_graph_construction=entity_graph_construction,
    community_clustering=community_clustering,
    report_generation=report_generation,
    chunker=chunker,
    passage_retrieval=passage_retrieval,
    qa=qa,
)
```

### Build the Index at Once:

```python
# Run all indexing steps
graphrag.make_index(
    documents=documents,
    index_dir="./indexes",
    retrieval_size=10,
    window_size=128,
    entity_dict=entity_dict,
    additional_triples=additional_triples,
    node_attr_keys=("name", "entity_type", "description"),
    edge_attr_keys=("relation",),
    passage_retrieval_indexing_kwargs={
        "batch_size": 1024,
    },
)
```

`retrieval_size` controls candidate retrieval during Entity Disambiguation.
`entity_dict` and `additional_triples` are passed to Entity Graph Construction.
`node_attr_keys` and `edge_attr_keys` select the graph attributes used during Report Generation.
`window_size` controls Chunking of community reports.
Values in `passage_retrieval_indexing_kwargs` are forwarded to the Passage Retrieval component. Omit arguments unsupported by the selected component.

### Build the Index Step by Step:

Set `target_component` to run only one indexing step.

```python
# Example 1: Run only Entity Graph Construction over the input under `index_dir`
graphrag.make_index(
    documents=documents,
    index_dir="./indexes",
    retrieval_size=10,
    window_size=128,
    entity_dict=entity_dict,
    additional_triples=additional_triples,
    target_component="entity_graph_construction",
)

# Example 2: Run only Report Generation over inputs under `index_dir`
graphrag.make_index(
    documents=documents,
    index_dir="./indexes",
    retrieval_size=10,
    window_size=128,
    target_component="report_generation",
)
```

When run individually, steps after NER load their inputs from the standard filenames under `index_dir`.

### Use the OpenAI Batch API During Indexing:

The NER, Entity Disambiguation (Reranking), and Document-level Relation Extraction steps support the OpenAI Batch API when their LLM component uses `OpenAILLM`.

```python
# Submit requests for NER
graphrag.make_index(
    documents=documents,
    index_dir="./indexes",
    retrieval_size=10,
    window_size=128,
    target_component="ner",
    batch_mode="submit",
    batch_dir="./batches",
)

# Fetch the responses for NER
graphrag.make_index(
    documents=documents,
    index_dir="./indexes",
    retrieval_size=10,
    window_size=128,
    target_component="ner",
    batch_mode="fetch",
    batch_dir="./batches",
)
```

When `batch_mode` is specified, `target_component` and `batch_dir` are required.
Valid modes are `"submit"` and `"fetch"`.
Running all indexing steps at once is not supported in batch mode.

### Load the Index:

```python
# Load the report-chunk retrieval index
graphrag.load_index(
    index_dir="./indexes",
)
```

`load_index()` must be called before inference.

### Run Inference:

```python
# Retrieve report chunks and answer multiple questions
result_questions = graphrag.infer(
    questions=questions,
    top_k=5,
)
```

`top_k` controls the number of report chunks returned by Passage Retrieval.

### Use the OpenAI Batch API During Inference:

When the Question Answering component is `LLMQA` backed by `OpenAILLM`, inference can submit and fetch answer-generation requests through the OpenAI Batch API.
Passage Retrieval still runs during both calls.

```python
# Submit requests for inference
result_questions = graphrag.infer(
    questions=questions,
    top_k=5,
    batch_mode="submit",
    batch_dir="./batches",
)
assert result_questions is None

# Fetch the responses for inference
result_questions = graphrag.infer(
    questions=questions,
    top_k=5,
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
| `documents_with_mentions.json` | Named Entity Recognition |
| `documents_with_candidates.json` | Entity Disambiguation (Retrieval) |
| `candidate_entities.json` | Entity Disambiguation (Retrieval) |
| `documents_with_entities.json` | Entity Disambiguation (Reranking) |
| `documents_with_triples.json` | Document-level Relation Extraction |
| `graph.graphml` | Entity Graph Construction |
| `communities.json` | Community Clustering |
| `reports.jsonl` | Report Generation |
| `chunked_reports.jsonl` | Chunking |
| `passage_retrieval_index/` | Passage Retrieval indexing |

The contents of `passage_retrieval_index/` depend on the selected Passage Retrieval component.

## Inference Output

Each output preserves the fields returned by the Question Answering component and adds `contexts`, containing the report chunks returned by Passage Retrieval.

The pipeline does not save inference output automatically. The caller is responsible for saving `result_questions`.

## Example

See [experiments/graphrag_pipeline_tacl2026](../../experiments/graphrag_pipeline_tacl2026) for runnable examples.
