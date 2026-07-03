# Chunking (`kapipe.chunking`)

**Chunking** splits a passage into smaller passages.

This component is useful for preparing long documents or generated reports for retrieval and question answering.

## Input

A passage is represented as a dictionary with the following fields.

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Passage title, if available |
| `text` | `str` | Passage text |

Additional metadata fields are preserved in the output.

```json
{
    "title": "Lithium Carbonate and Related Health Conditions",
    "text": "This report examines the interconnections between Lithium Carbonate, ...",
    "hoge": "fuga"
}
```

## Output

The output is a list of chunked passages.

Each chunked passage preserves the original metadata and replaces `text` with a shorter text chunk.

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Original passage title, if available |
| `text` | `str` | Chunk text |

```json
[
    {
        "title": "Lithium Carbonate and Related Health Conditions",
        "text": "This report examines the interconnections between Lithium Carbonate, ...",
        "hoge": "fuga"
    },
    {
        "title": "Lithium Carbonate and Related Health Conditions",
        "text": "This duality necessitates careful monitoring of patients receiving Lithium treatment, ...",
        "hoge": "fuga"
    },
    ...
]
```

## Supported Methods

| Method | Description |
|---|---|
| Chunker | Splits text into sentence-based chunks with a maximum word window |

## Usage

### Default Chunking:

```python
from kapipe.chunking import Chunker

# Build a Chunker with the default English sentencizer
chunker = Chunker()

# Split a passage into chunked passages
chunked_passages = chunker.split_passage_to_chunked_passages(
    passage=passage,
    window_size=100,
)
```

### Chunking with a spaCy Model:

```python
from kapipe.chunking import Chunker

# Build a chunker with a spaCy model
chunker = Chunker(
    model_name="en_core_web_sm",
)

# Split a passage into chunked passages
chunked_passages = chunker.split_passage_to_chunked_passages(
    passage=passage,
    window_size=100,
)
```

## Metadata Preservation

Chunking preserves metadata fields other than `title` and `text`.

This is useful when chunking community reports, because fields such as `community_id`, `nodes`, and `level` remain attached to each chunk.

## Example

See [experiments/chunking](../../experiments/chunking) for runnable examples.
