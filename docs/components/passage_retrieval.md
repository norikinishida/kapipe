# Passage Retrieval (`kapipe.passage_retrieval`)

**Passage Retrieval** retrieves relevant passages for a query.

This component first builds an index over passages, then searches the index for each query.

## Input

Indexing takes a list of passages.

Each passage contains the following fields.

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Passage title, if available |
| `text` | `str` | Passage text |

Additional metadata fields are preserved in retrieved passages.

```json
[
    {
        "title": "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant.",
        "text": "A newborn with massive tricuspid regurgitation, atrial flutter, congestive heart failure, and a high serum lithium level is described. This is the first patient to initially manifest tricuspid regurgitation and atrial flutter, and the 11th described patient with cardiac disease among infants exposed to lithium compounds in the first trimester of pregnancy. Sixty-three percent of these infants had tricuspid valve involvement. Lithium carbonate may be a factor in the increasing incidence of congenital heart disease when taken during early pregnancy. It also causes neurologic depression, cyanosis, and cardiac arrhythmia when consumed prior to delivery."
    },
    ...
]
```

Search takes a list of query strings.

```json
[
    "Which interventions attenuated, prevented, antagonized, or reduced opioid-induced muscle rigidity in rats?",
]
```

## Output

The output is a list of retrieved passages for each query.

Each retrieved passage preserves the original passage fields and adds `score` and `rank`.

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Passage title, if available |
| `text` | `str` | Passage text |
| `score` | `float` | Retrieval score |
| `rank` | `int` | One-based rank |

```json
[
    [
        {
            "title": "Ketanserin pretreatment reverses alfentanil-induced muscle rigidity.",
            "text": "Systemic pretreatment with ketanserin, a relatively specific type-2 serotonin receptor antagonist, ...",
            "score": 0.7015140056610107,
            "rank": 1
        },
        {
            "title": "Involvement of locus coeruleus and noradrenergic neurotransmission in fentanyl-induced muscular rigidity in the rat.",
            "text": "Whereas muscular rigidity is a well-known side effect that is associated with high-dose fentanyl anesthesia, ...",
            "score": 0.6968926191329956,
            "rank": 2
        },
        ...
    ],
    ...
]
```

## Supported Methods

| Method | Description |
|---|---|
| BM25 | Sparse lexical retrieval based on term frequency and inverse document frequency |
| [Contriever (Izacard et al., 2022)](https://arxiv.org/abs/2112.09118) | Dense passage retrieval with a dual-encoder model |
| Qwen3-Embedding | Dense passage retrieval with Qwen3 embedding models |

## Usage

### BM25:

```python
from kapipe.passage_retrieval import BM25

# Define a simple tokenizer
def tokenizer(text: str) -> list[str]:
    return text.lower().split()

# Instantiate a BM25-based Passage Retrieval component
retriever = BM25(
    tokenizer=tokenizer,
)

# Build an index over passages
retriever.make_index(
    passages=passages,
    index_dir="/path/to/index",
)

# Retrieve the top-10 passages for each query
retrieved_passages = retriever.search(
    queries=[question["question"]],
    top_k=10,
)[0]
```

### Contriever:

```python
from kapipe.passage_retrieval import Contriever

# Instantiate a Contriever-based Passage Retrieval component
retriever = Contriever(
    model_name="facebook/contriever-msmarco",
    max_passage_length=512,
    pooling_method="average",
    normalize=False,
    metric="inner-product",
)

# Build and save an index over passages
retriever.make_index(
    passages=passages,
    index_dir="/path/to/index",
    batch_size=64,
)

# Retrieve the top-10 passages for each query
retrieved_passages = retriever.search(
    queries=[question["question"]],
    top_k=10,
)[0]
```

### Qwen3-Embedding:

```python
from kapipe.passage_retrieval import Qwen3Embedding

# Instantiate a Qwen3-Embedding-based Passage Retrieval component
retriever = Qwen3Embedding(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    max_passage_length=8192,
    normalize=True,
    metric="inner-product",
    query_instruction="Given a question, retrieve relevant passages that answer the question.",
)

# Build and save an index over passages
retriever.make_index(
    passages=passages,
    index_dir="/path/to/index",
    batch_size=8,
)

# Retrieve the top-10 passages for each query
retrieved_passages = retriever.search(
    queries=[question["question"]],
    top_k=10,
)[0]
```

After building an index, you can reload it with `retriever.load_index(index_dir="/path/to/index")`.

## Indexing Custom Passages

You can index any passage list whose items contain a `text` field.

If a passage has a `title`, the title is used together with the text during retrieval.

## Example

See [experiments/passage_retrieval](../../experiments/passage_retrieval) for runnable examples.
