# GraphRAG Pipeline (`kapipe.pipelines.GraphRAGPipeline`)

**GraphRAG Pipeline** connects components for triple extraction, knowledge graph organization, retrieval, and question answering.

Each step can be run independently. Intermediate results are saved under `index_dir`, allowing later steps to resume from previously generated files.

The pipeline does not define the behavior of the individual components. See the corresponding component documentation for their inputs, outputs, methods, and configuration.

## Component Flow

Knowledge extraction and structuring:

```text
Documents (input)
  → Named Entity Recognition
  → Entity Disambiguation (Retrieval)
  → Entity Disambiguation (Reranking)
  → Document-level Relation Extraction
  → Entity Graph Construction
  → Community Clustering
  → Report Generation
  → Chunking
  → Passage Retrieval index (output)
```

Inference:

```text
Question (input)
  → Passage Retrieval
  → Question Answering
  → Answer (output)
```

Raw text can optionally be converted into a document using the Chunking component before knowledge structuring.

## Components

| Constructor Argument | Component |
|---|---|
| `ner` | [Named Entity Recognition](../components/ner.md) |
| `ed_retrieval` | [Entity Disambiguation (Retrieval)](../components/ed_retrieval.md) |
| `ed_reranking` | [Entity Disambiguation (Reranking)](../components/ed_reranking.md) |
| `docre` | [Document-level Relation Extraction](../components/docre.md) |
| `entity_graph_construction` | [Entity Graph Construction](../components/entity_graph_construction.md) |
| `community_clustering` | [Community Clustering](../components/community_clustering.md) |
| `report_generation` | [Report Generation](../components/report_generation.md) |
| `chunker` | [Chunking](../components/chunking.md) |
| `passage_retrieval` | [Passage Retrieval](../components/passage_retrieval.md) |
| `qa` | [Question Answering](../components/qa.md) |

All constructor arguments must be specified. Components unused by the intended step may be set to `None`.

Calling a processing method without its required component raises `ValueError`.

## Methods

The pipeline provides two high-level methods.

| Method | Description |
|---|---|
| `make_index()` | Runs all indexing steps and builds a retrieval index over community report chunks |
| `infer()` | Retrieves report chunks for one question and generates an answer |

The indexing steps can also be run independently through the following component-level methods.

| Step | Method | Connected Component |
|---|---|---|
| Optional | `convert_text_to_document()` | Chunking |
| 1 | `extract_triples()` | NER, ED Retrieval, ED Reranking, DocRE |
| 2 | `construct_entity_graph()` | Entity Graph Construction |
| 3 | `cluster_communities()` | Community Clustering |
| 4 | `generate_community_reports()` | Report Generation |
| 5 | `chunk_reports()` | Chunking |
| 6 | `make_passage_retrieval_index()` | Passage Retrieval |

The following methods load intermediate results.

| Method | Loaded Result |
|---|---|
| `load_documents_with_triples()` | Documents with extracted triples |
| `load_entity_graph()` | Entity graph |
| `load_communities()` | Community records |
| `load_community_reports()` | Community reports |
| `load_chunked_reports()` | Chunked community reports |
| `load_passage_retrieval_index()` | Passage retrieval index |

## Usage

### Initialize the Pipeline

```python
from kapipe.pipelines import GraphRAGPipeline


# Initialize the components before constructing the pipeline
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

# Connect the initialized components
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

### Build the Index

```python
# Set the shared directory for intermediate results
index_dir = "./indexes"

# Run all indexing steps
graphrag.make_index(
    documents=documents,
    retrieval_size=10,
    window_size=128,
    index_dir=index_dir,
    entity_dict_path="./entity_dict.json",
    additional_triples_path=None,
    passage_retrieval_indexing_kwargs={
        "batch_size": 64,
    },
)
```

Values in `passage_retrieval_indexing_kwargs` are forwarded to the Passage Retrieval component. Omit arguments unsupported by the selected component.

### Run Inference

```python
# Retrieve report chunks from the index and answer one question
result_question = graphrag.infer(
    question=question,
    top_k=5,
)
```

The retrieved report chunks are preserved in the pipeline output as `contexts`.

### Run Individual Indexing Steps

Each indexing step remains available separately for inspection, replacement, or resumption.

```python
import os


# Extract triples from the input documents
documents_with_triples = graphrag.extract_triples(
    documents=documents,
    retrieval_size=10,
    index_dir=index_dir,
)

# Construct an entity graph from the saved triple extraction result
graph = graphrag.construct_entity_graph(
    documents_path_list=[
        os.path.join(index_dir, "documents_with_triples.json"),
    ],
    entity_dict_path="./entity_dict.json",
    additional_triples_path=None,
    index_dir=index_dir,
)

# Cluster the entity graph into communities
communities = graphrag.cluster_communities(
    graph=graph,
    index_dir=index_dir,
)

# Generate textual reports from the communities
reports = graphrag.generate_community_reports(
    graph=graph,
    communities=communities,
    index_dir=index_dir,
)

# Split the reports into chunks for retrieval
chunked_reports = graphrag.chunk_reports(
    reports=reports,
    window_size=128,
    index_dir=index_dir,
)

# Build a retrieval index over the report chunks
graphrag.make_passage_retrieval_index(
    chunked_reports=chunked_reports,
    index_dir=index_dir,
    batch_size=64,
)
```

Additional keyword arguments passed to `make_passage_retrieval_index()` are forwarded to the Passage Retrieval component.

### Resume from Intermediate Results

```python
# Load the entity graph and communities from previous steps
graph = graphrag.load_entity_graph(
    index_dir="./indexes",
)
communities = graphrag.load_communities(
    index_dir="./indexes",
)

# Resume the pipeline from report generation
reports = graphrag.generate_community_reports(
    graph=graph,
    communities=communities,
    index_dir="./indexes",
)
```

A pipeline used for only a subset of the steps may set the other components to `None`.

```python
from kapipe.pipelines import GraphRAGPipeline


# Initialize only the components required for inference
passage_retrieval = ...
qa = ...

# Omit components that are not used during inference
graphrag = GraphRAGPipeline(
    ner=None,
    ed_retrieval=None,
    ed_reranking=None,
    docre=None,
    entity_graph_construction=None,
    community_clustering=None,
    report_generation=None,
    chunker=None,
    passage_retrieval=passage_retrieval,
    qa=qa,
)

# Load the index and run inference
graphrag.load_passage_retrieval_index(
    index_dir="./indexes",
)
result_question = graphrag.infer(
    question=question,
    top_k=5,
)
```

## Intermediate Files

`make_index()` creates all intermediate results and the component-dependent retrieval index listed below.

| File | Created By | Loaded By |
|---|---|---|
| `documents_with_triples.json` | `extract_triples()` | `load_documents_with_triples()` |
| `graph.graphml` | `construct_entity_graph()` | `load_entity_graph()` |
| `communities.json` | `cluster_communities()` | `load_communities()` |
| `reports.jsonl` | `generate_community_reports()` | `load_community_reports()` |
| `chunked_reports.jsonl` | `chunk_reports()` | `load_chunked_reports()` |
| Component-dependent index files | `make_passage_retrieval_index()` | `load_passage_retrieval_index()` |

All fixed intermediate filenames are created under `index_dir`.

The Passage Retrieval index format and filenames depend on the selected Passage Retrieval component.

## Example

See [experiments/graphrag_pipeline_tacl2026](../../experiments/graphrag_pipeline_tacl2026) for a runnable example.
