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
    "title": "Oliver Hartwell and Clara Venn",
    "text": "Oliver Hartwell married Clara Venn in 2016. Clara was known in Hartwell's family for her quiet humor and careful memory. Public records in Larkford list Clara Venn as Oliver Hartwell's wife. Clara Venn began her clinical career at St. Brigid's Children's Hospital. She worked there as a pediatric nurse from 2012 to 2016. Her duties included night rounds and discharge planning. Clara Venn later joined North Quay Medical Center. She worked in its pediatric ward from 2017 to 2021. The hospital newsletter described her as a senior nurse. Clara Venn attended a short training course at Alderwick General Hospital. The course focused on emergency triage for children. She did not hold a staff position there.",
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
        "title": "Oliver Hartwell and Clara Venn",
        "text": "Oliver Hartwell married Clara Venn in 2016. Clara was known in Hartwell's family for her quiet humor and careful memory. Public records in Larkford list Clara Venn as Oliver Hartwell's wife. Clara Venn began her clinical career at St. Brigid's Children's Hospital. She worked there as a pediatric nurse from 2012 to 2016.",
        "hoge": "fuga"
    },
    {
        "title": "Oliver Hartwell and Clara Venn",
        "text": "Her duties included night rounds and discharge planning. Clara Venn later joined North Quay Medical Center. She worked in its pediatric ward from 2017 to 2021. The hospital newsletter described her as a senior nurse. Clara Venn attended a short training course at Alderwick General Hospital. The course focused on emergency triage for children.",
        "hoge": "fuga"
    },
    {
        "title": "Oliver Hartwell and Clara Venn",
        "text": "She did not hold a staff position there.",
        "hoge": "fuga"
    }
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

This is useful when chunking community reports, because fields such as `source`, `publication_date`, `community_id`, and `nodes` remain attached to each chunk.

## Example

See [experiments/chunking](../../experiments/chunking) for runnable examples.
