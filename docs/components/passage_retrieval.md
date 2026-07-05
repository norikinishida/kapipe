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
        "title": "Effect of calcium chloride and 4-aminopyridine therapy on desipramine toxicity in rats.",
        "text": "BACKGROUND: Hypotension is a major contributor to mortality in tricyclic antidepressant overdose. Recent data suggest that ..."
    },
    ...
]
```

Search takes a list of query strings.

```json
[
    "Which metastatic breast cancer treatments were linked to an adverse event also reported in dogs treated with lomustine?",
    ...
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
            "title": "CCNU (lomustine) toxicity in dogs: a retrospective study (2002-07).",
            "text": "OBJECTIVE: To describe the incidence of haematological, renal, hepatic and gastrointestinal toxicities in tumour-bearing dogs receiving 1-(2-chloroethyl)-3-cyclohexyl-1-nitrosourea (CCNU). DESIGN: The medical records of 206 dogs that were treated with CCNU at the Melbourne Veterinary Specialist Centre between February 2002 and December 2007 were retrospectively evaluated. RESULTS: Of the 206 dogs treated with CCNU, 185 met the inclusion criteria for at least one class of toxicity. CCNU was used most commonly in the treatment of lymphoma, mast cell tumour, brain tumour, histiocytic tumours and epitheliotropic lymphoma. Throughout treatment, 56.9% of dogs experienced neutropenia, 34.2% experienced anaemia and 14.2% experienced thrombocytopenia. Gastrointestinal toxicosis was detected in 37.8% of dogs, the most common sign of which was vomiting (24.3%). Potential renal toxicity and elevated alanine transaminase (ALT) concentration were reported in 12.2% and 48.8% of dogs, respectively. The incidence of hepatic failure was 1.2%. CONCLUSIONS: CCNU-associated toxicity in dogs is common, but is usually not life threatening.",
            "score": 0.7051769495010376,
            "rank": 1
        },
        {
            "title": "Reduced cardiotoxicity and preserved antitumor efficacy of liposome-encapsulated doxorubicin and cyclophosphamide compared with conventional doxorubicin and cyclophosphamide in a randomized, multicenter trial of metastatic breast cancer.",
            "text": "PURPOSE: To determine whether Myocet (liposome-encapsulated doxorubicin; The Liposome Company, Elan Corporation, Princeton, NJ) in combination with cyclophosphamide significantly reduces doxorubicin cardiotoxicity while providing comparable antitumor efficacy in first-line treatment of metastatic breast cancer (MBC). PATIENTS AND METHODS: Two hundred ninety-seven patients with MBC and no prior chemotherapy for metastatic disease were randomized to receive either 60 mg/m (2) of Myocet (M) or conventional doxorubicin (A), in combination with 600 mg/m (2) of cyclophosphamide (C), every 3 weeks until disease progression or unacceptable toxicity. Cardiotoxicity was defined by reductions in left-ventricular ejection fraction, assessed by serial multigated radionuclide angiography scans, or congestive heart failure (CHF). Antitumor efficacy was assessed by objective tumor response rates (World Health Organization criteria), time to progression, and survival. RESULTS: Six percent of MC patients versus 21% (including five cases of CHF) of AC patients developed cardiotoxicity (P = . 0002). Median cumulative doxorubicin dose at onset was more than 2,220 mg/m (2) for MC versus 480 mg/m (2) for AC (P = . 0001, hazard ratio, 5.04). MC patients also experienced less grade 4 neutropenia. Antitumor efficacy of MC versus AC was comparable: objective response rates, 43% versus 43%; median time to progression, 5.1% versus 5.5 months; median time to treatment failure, 4.6 versus 4.4 months; and median survival, 19 versus 16 months. CONCLUSION: Myocet improves the therapeutic index of doxorubicin by significantly reducing cardiotoxicity and grade 4 neutropenia and provides comparable antitumor efficacy, when used in combination with cyclophosphamide as first-line therapy for MBC.",
            "score": 0.5899654626846313,
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

# Instantiate the BM25-based Passage Retrieval component
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

# Instantiate the Contriever-based Passage Retrieval component
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

# Instantiate the Qwen3-Embedding-based Passage Retrieval component
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
