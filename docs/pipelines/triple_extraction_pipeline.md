# Triple Extraction Pipeline (`kapipe.pipelines.TripleExtractionPipeline`)

**Triple Extraction Pipeline** connects components that extract relational triples from documents.

Each extraction step can be run independently. Intermediate results are saved under `intermediate_dir`, allowing later steps to resume from previously generated files.

The pipeline does not define the behavior of the individual components. See the corresponding component documentation for their inputs, outputs, methods, and configuration.

## Component Flow

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
```

Raw text can be converted into a document before triple extraction.

```text
Raw text (input)
↓ Chunking
Document (output)
```

## Components

| Constructor Argument | Component |
|---|---|
| `ner` | [Named Entity Recognition](../components/ner.md) |
| `ed_retrieval` | [Entity Disambiguation (Retrieval)](../components/ed_retrieval.md) |
| `ed_reranking` | [Entity Disambiguation (Reranking)](../components/ed_reranking.md) |
| `docre` | [Document-level Relation Extraction](../components/docre.md) |
| `chunker` | [Chunking](../components/chunking.md) |

All constructor arguments must be specified.

## Pipeline Methods

| Method | Description |
|---|---|
| `convert_text_to_document()` | Converts raw text into a document using the Chunking component |
| `extract_triples()` | Runs all triple extraction steps or one selected extraction step |

## Usage

### Initialize the Pipeline:

```python
from kapipe.pipelines import TripleExtractionPipeline

# Initialize the components
ner = ...
ed_retrieval = ...
ed_reranking = ...
docre = ...
chunker = ...

# Instantiate the Triple Extraction pipeline
triple_extraction = TripleExtractionPipeline(
    ner=ner,
    ed_retrieval=ed_retrieval,
    ed_reranking=ed_reranking,
    docre=docre,
    chunker=chunker,
)
```

### Convert Raw Text into a Document:

```python
# Convert raw text into the document format
document = triple_extraction.convert_text_to_document(
    doc_key="document#001",
    text="Aspirin may reduce fever.",
    title="Example document",
)
```

### Extract Triples at Once:

```python
# Run all triple extraction steps
result_documents = triple_extraction.extract_triples(
    documents=documents,
    retrieval_size=10,
)
```

`retrieval_size` controls candidate retrieval during Entity Disambiguation.

### Extract Triples Step by Step:

Set `target_component` to run only one extraction step.

```python
# Example 1: Run only Named Entity Recognition
triple_extraction.extract_triples(
    documents=documents,
    retrieval_size=10,
    target_component="ner",
    intermediate_dir="./intermediate",
)

# Example 2: Run only Entity Disambiguation (Retrieval) over the input under `intermediate_dir`
triple_extraction.extract_triples(
    documents=documents,
    retrieval_size=10,
    target_component="ed_retrieval",
    intermediate_dir="./intermediate",
)
```

When run individually, steps after Named Entity Recognition load their inputs from the standard filenames under `intermediate_dir`.

### Use the OpenAI Batch API:

The Named Entity Recognition, Entity Disambiguation (Reranking), and Document-level Relation Extraction steps support the OpenAI Batch API when their LLM component uses `OpenAILLM`.

```python
# Submit requests for Named Entity Recognition
triple_extraction.extract_triples(
    documents=documents,
    retrieval_size=10,
    target_component="ner",
    intermediate_dir="./intermediate",
    batch_mode="submit",
    batch_dir="./batches",
)

# Fetch the responses for Named Entity Recognition
triple_extraction.extract_triples(
    documents=documents,
    retrieval_size=10,
    target_component="ner",
    intermediate_dir="./intermediate",
    batch_mode="fetch",
    batch_dir="./batches",
)
```

When `batch_mode` is specified, `target_component`, `intermediate_dir`, and `batch_dir` are required.
Valid modes are `"submit"` and `"fetch"`.
Running all extraction steps at once is not supported in batch mode.

## Intermediate Outputs

When `target_component` is specified, `extract_triples()` creates the following artifacts under `intermediate_dir`.

| Path | Created By |
|---|---|
| `documents_with_mentions.json` | Named Entity Recognition |
| `documents_with_candidates.json` | Entity Disambiguation (Retrieval) |
| `candidate_entities.json` | Entity Disambiguation (Retrieval) |
| `documents_with_entities.json` | Entity Disambiguation (Reranking) |

## Triple Extraction Output

Running all extraction steps or Document-level Relation Extraction returns the documents with extracted triples.
The other individual extraction steps and Batch API submission return `None`.

The pipeline does not save the documents with extracted triples automatically. The caller is responsible for saving `result_documents`.

## Example

See [experiments/triple_extraction_pipeline](../../experiments/triple_extraction_pipeline) for a runnable example.
