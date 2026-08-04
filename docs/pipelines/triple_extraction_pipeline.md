# Triple Extraction Pipeline (`kapipe.pipelines.TripleExtractionPipeline`)

**Triple Extraction Pipeline** connects components that extract relational triples from a document.

The pipeline does not define the behavior of the individual components. See the corresponding component documentation for their inputs, outputs, methods, and configuration.

## Component Flow

```text
Document
  → Named Entity Recognition
  → Entity Disambiguation (Retrieval)
  → Entity Disambiguation (Reranking)
  → Document-level Relation Extraction
  → Document with triples
```

Raw text can optionally be converted into a document before triple extraction.

```text
Raw text
  → Chunking
  → Document
```

## Components

| Constructor Argument | Component | Required |
|---|---|---|
| `ner` | [Named Entity Recognition](../components/ner.md) | Yes |
| `ed_retrieval` | [Entity Disambiguation (Retrieval)](../components/ed_retrieval.md) | Yes |
| `ed_reranking` | [Entity Disambiguation (Reranking)](../components/ed_reranking.md) | Yes |
| `docre` | [Document-level Relation Extraction](../components/docre.md) | Yes |
| `chunker` | [Chunking](../components/chunking.md) | Only for raw-text conversion |

The components must be instantiated before they are passed to the pipeline.

## Methods

| Method | Description |
|---|---|
| `convert_text_to_document()` | Converts raw text into a document using the Chunking component |
| `extract_triples()` | Applies NER, ED Retrieval, ED Reranking, and DocRE to one document |

### `convert_text_to_document()`

```python
document = pipeline.convert_text_to_document(
    doc_key="document#001",
    text="Aspirin may reduce fever.",
    title="Example document",
)
```

This method requires the optional `chunker` component.

### `extract_triples()`

```python
result_document = pipeline.extract_triples(
    document=document,
    retrieval_size=10,
)
```

`retrieval_size` is passed to the ED Retrieval component.

## Usage

```python
from kapipe.pipelines import TripleExtractionPipeline


# Initialize the components before constructing the pipeline
chunker = ...
ner = ...
ed_retrieval = ...
ed_reranking = ...
docre = ...

# Connect the initialized components
pipeline = TripleExtractionPipeline(
    chunker=chunker,
    ner=ner,
    ed_retrieval=ed_retrieval,
    ed_reranking=ed_reranking,
    docre=docre,
)

# Convert raw text into the document format
document = pipeline.convert_text_to_document(
    doc_key="document#001",
    text="Aspirin may reduce fever.",
    title="Example document",
)

# Apply the connected triple extraction components
result_document = pipeline.extract_triples(
    document=document,
    retrieval_size=10,
)
```

If the input is already represented as a document, `chunker` can be omitted.

```python
from kapipe.pipelines import TripleExtractionPipeline


# Initialize the required triple extraction components
ner = ...
ed_retrieval = ...
ed_reranking = ...
docre = ...

# Connect the components without a Chunking component
pipeline = TripleExtractionPipeline(
    ner=ner,
    ed_retrieval=ed_retrieval,
    ed_reranking=ed_reranking,
    docre=docre,
)

# Apply the pipeline to an existing document
result_document = pipeline.extract_triples(
    document=document,
    retrieval_size=10,
)
```

## Intermediate Files

`TripleExtractionPipeline` does not save intermediate results automatically.

The caller is responsible for loading input documents and saving the resulting documents.

## Example

See [experiments/triple_extraction_pipeline](../../experiments/triple_extraction_pipeline) for a runnable example.