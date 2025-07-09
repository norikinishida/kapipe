# KAPipe: A Modular Pipeline for Knowledge Acquisition

## Table of Contents

- [🤖 What is KAPipe?](#what-is-kapipe)
- [📦 Installation](#-installation)
- [🧩 Triple Extraction](#-triple-extraction)
- [🕸️ Knowledge Graph Construction](#-knowledge-graph-construction)
- [🧱 Community Clustering](#-community-clustering)
- [📝 Report Generation](#-report-generation)
- [✂️ Chunking](#-chunking)
- [🔍 Passage Retrieval](#-passage-retrieval)
- [💬 Question Answering](#-question-answering)

## 🤖 What is KAPipe?

**KAPipe** is a modular pipeline for comprehensive **knowledge acquisition** from unstructured documents.  
It supports **extraction**, **organization**, **retrieval**, and **utilization** of knowledge, serving as a core framework for building intelligent systems that reason over structured knowledge.  

Currently, KAPipe provides the following functionalities:

- 🧩**Triple Extraction**  
    - Extract facts in the form of (head entity, relation, tail entity) triples from raw text.

- 🕸️**Knowledge Graph Construction**  
    - Build a symbolic knowledge graph from triples, optionally augmented with external ontologies or knowledge bases (e.g., Wikidata, UMLS).

- 🧱**Community Clustering**  
    - Cluster the knowledge graph into semantically coherent subgraphs (*communities*).

- 📝**Report Generation**  
    - Generate textual reports (or summaries) of graph communities.

- ✂️**Chunking**  
    - Split text (e.g., community report) into fixed-size chunks based on a predefined token length (e.g., n=300).

- 🔍**Passage Retrieval**  
    - Retrieve relevant chunks for given queries using lexical or dense retrieval.

- 💬**Question Answering**  
    - Answer questions using retrieved chunks as context.

These components together form an implementation of **graph-based retrieval-augmented generation (GraphRAG)**, enabling question answering and reasoning grounded in structured knowledge.

## 📦 Installation

### Step 1: Set up a Python environment
```bash
python -m venv .env
source .env/bin/activate
pip install -U pip setuptools wheel
```

### Step 2: Install KAPipe
```bash
pip install kapipe
```

### Step 3: Download pretrained models and configurations

Pretrained models and configuration files can be downloaded from the following Google Drive folder:

📁 [KAPipe Release Files](https://drive.google.com/drive/folders/16ypMCoLYf5kDxglDD_NYoCNAfhTy4Qwp)

Download the latest release file named release.YYYYMMDD.tar.gz, then extract it to the ~/.kapipe directory:

```bash
mkdir -p ~/.kapipe
mv release.YYYYMMDD.tar.gz ~/.kapipe
cd ~/.kapipe
tar -zxvf release.YYYYMMDD.tar.gz
```

## 🧩 Triple Extraction

### Overview

The **Triple Extraction** module identifies relational facts from raw text in the form of (head entity, relation, tail entity) **triples**.

Specifically, this is achieved through the following cascade of subtasks:

1. **Named Entity Recognition (NER):**
    - Detect entity mentions (spans) and classify their types.
1. **Entity Disambiguation Retrieval (ED-Retrieval)**:
    - Retrieve candidate concept IDs from a knowledge base for each mention.
1. **Entity Disambiguation Reranking (ED-Reranking)**:
    - Select the most probable concept ID from the retrieved candidates.
1. **Document-level Relation Extraction (DocRE)**:
    - Extract relational triples based on the disambiguated entity set.

### Input

This module takes as input:

1. ***Document***, or a dictionary containing
    - `doc_key` (str): Unique identifier for the document
    - `sentences` (list[str]): List of sentence strings (tokenized)

```json
{
    "doc_key": "6794356",
    "sentences": [
        "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant .",
        "A newborn with massive tricuspid regurgitation , atrial flutter , congestive heart failure , and a high serum lithium level is described .",
        ...
    ]
}
```
(See `experiments/data/examples/documents_without_triples.json` for more details.)

Each subtask takes a ***Document*** object as input, augments it with new fields, and returns it.  
This allows custom metadata to persist throughout the pipeline.

### Output

The output is also the same-format dictionary (***Document***), augmented with extracted entities and triples information:

- `doc_key` (str): Same as input
- `sentences` (list[str]): Same as input
- `mentions` (list[dict]): Mentions, or a list of dictionaries, each containing:
    - `span` (tuple[int,int]): Mention span
    - `name` (str): Mention string
    - `entity_type` (str): Entity type
    - `entity_id` (str): Concept ID
- `entities` (list[dict]): Entities, or a list of dictionaries, each containing
    - `mention_indices` (list[int]): Indices of mentions belonging to this entity
    - `entity_type` (str): Entity type 
    - `entity_id` (str): Concept ID
- `relations` (list[dict]): Triples, or a list dictionaries, each containing
    - `arg1` (int): Index of the head/subject entity
    - `relation` (str): Semantic relation
    - `arg2` (int): Index of the tail/object entity

```json
{
    "doc_key": "6794356",
    "sentences": [...],
    "mentions": [
        {
            "span": [0, 2],
            "name": "Tricuspid valve regurgitation",
            "entity_type": "Disease",
            "entity_id": "D014262"
        },
        ...
    ],
    "entities": [
        {
            "mention_indices": [0, 3, 7],
            "entity_type": "Disease",
            "entity_id": "D014262"
        },
        ...
    ],
    "relations": [
        {
            "arg1": 1,
            "relation": "CID",
            "arg2": 7
        },
        ...
    ]
}
```
(See `experiments/data/examples/documents_with_triples.json` for more details.)

### How to Use

```python
import kapipe.triple_extraction

IDENTIFIER = "biaffinener_blink_blink_atlop_cdr"

# Load the Triple Extraction pipeline
pipe = kapipe.triple_extraction.load(
    identifier=IDENTIFIER,
    gpu_map={"ner": 0, "ed_retrieval": 0, "ed_reranking": 2, "docre": 3}
)

# We provide a utility for converting text (string) to Document format
# `title` is optional
document = pipe.text_to_document(doc_key=your_doc_key, text=your_text, title=your_title)

# Apply the pipeline to your input document
document = pipe(document)
```
(See `experiments/codes/run_triple_extraction.py` for specific examples.)

<!-- The `identifier` determines the specific models used for each subtask.  
For example, `"biaffinener_blink_blink_atlop_cdr"` uses:

- **NER**: Biaffine-NER (trained on BC5CDR for Chemical and Disease types)
- **ED-Retrieval**: BLINK Bi-Encoder (trained on BC5CDR for MeSH 2015)
- **ED-Reranking**: BLINK Cross-Encoder (trained on BC5CDR for MeSH 2015)
- **DocRE**: ATLOP (trained on BC5CDR for Chemical-Induce-Disease (CID) relation) -->

### Supported Methods

#### Named Entity Recognition (NER)
- **Biaffine-NER** ([`Yu et al., 2020`](https://aclanthology.org/2020.acl-main.577/)): Span-based BERT model using biaffine scoring
- **LLM-NER**: A proprietary/open-source LLM using a NER-specific prompt template and few-shot examples

#### Entity Disambiguation Retrieval (ED-Retrieval)
- **BLINK Bi-Encoder** ([`Wu et al., 2020`](https://aclanthology.org/2020.emnlp-main.519/)): Dense retriever using BERT-based encoders and approximate nearest neighbor search
- **BM25**: Lexical matching
- **Levenshtein**: Edit distance matching

#### Entity Disambiguation Reranking (ED-Reranking)
- **BLINK Cross-Encoder** (Wu et al., 2020): Reranker using a BERT-based encoder for candidates from the Bi-Encoder
- **LLM-ED**: A proprietary/open-source LLM using a ED-specific prompt template and few-shot examples

#### Document-level Relation Extraction (DocRE)
- **ATLOP** ([`Zhou et al., 2021`](https://ojs.aaai.org/index.php/AAAI/article/view/17717)): BERT-based model for DocRE
- **MA-ATLOP** (Oumaima & Nishida et al., 2024): Mention-agnostic extension of ATLOP
- **MA-QA** (Oumaima & Nishida, 2024): Question-answering style DocRE model
- **LLM-DocRE**: A proprietary/open-source LLM using a DocRE-specific prompt template and few-shot examples

### Available Pipeline Identifiers

The following pipeline configurations are currently available:

| identifier | NER (Entity Types) | ED-Retrieval (Knowledge Base) | ED-Reranking (Knowledge Base) | DocRE (Relations) |
| --- | --- | --- | --- | --- |
| `biaffinener_blink_blink_atlop_cdr` | Biaffine-NER ({Chemical, Disease}) | BLINK Bi-Encoder (MeSH 2015) | BLINK Cross-Encoder (MeSH 2015) | ATLOP ({Chemical-Induce-Disease}) |
| `biaffinener_blink_blink_atlop_linked_docred` | Biaffine-NER ({Person, Organization, Location, Time, Number, Misc}) | BLINK Bi-Encoder (DBPedia 2020.02.01) | BLINK Cross-Encoder (DBPedia 2020.02.01) | ATLOP (DBPedia 96 relations) |
| `llmner_blink_llmed_llmdocre_cdr` | LLM-NER `gpt-4o-mini` ({Chemical, Disease}) | BLINK Bi-Encoder (MeSH 2015) | LLM-ED `gpt-4o-mini` (MeSH 2015) | LLM-DocRE `gpt-4o-mini` ({Chemical-Induce-Disease}) |
| `llmner_blink_llmed_llmdocre_linked_docred` | LLM-NER `gpt-4o-mini` ({Person, Organization, Location, Time, Number, Misc}) | BLINK Bi-Encoder (DBPedia 2020.02.01) | LLM-ED `gpt-4o-mini` (DBPedia 2020.02.01) | LLM-DocRE `gpt-4o-mini` (DBPedia 96 relations) |

## 🕸️ Knowledge Graph Construction

### Overview

The **Knowledge Graph Construction** module builds a **directed multi-relational graph** from a set of extracted triples.

- **Nodes** represent unique entities (i.e., concepts).
- **Edges** represent semantic relations between entities.

### Input

This module takes as input:

1. List of ***Document*** objects with triples, produced by the **Triple Extraction** module

2. (optional) ***Additional Triples*** (existing KBs), or a list of dictionaries, each containing:
    - `head` (str): Entity ID of the subject
    - `relation` (str): Relation type (e.g., treats, causes)
    - `tail` (str): Entity ID of the object
```json
[
    {
        "head": "D000001",
        "relation": "treats",
        "tail": "D014262"
    },
    ...
]
```
(See `experiments/data/examples/additional_triples.json` for more details.)


3. ***Entity Dictionary***, or a list of dictionaries, each containing:
    - `entity_id` (str): Unique concept ID
    - `canonical_name` (str): Official name of the concept
    - `entity_type` (str): Type/category of the concept
    - `synonyms` (list[str]): A list of alternative names
    - `description` (str): Textual definition of the concept
```JSON
[
    {
        "entity_id": "C009166",
        "entity_index": 252696,
        "entity_type": null,
        "canonical_name": "retinol acetate",
        "synonyms": [
            "retinyl acetate",
            "vitamin A acetate"
        ],
        "description": "",
        "tree_numbers": []
    },
    {
        "entity_id": "D000641",
        "entity_index": 610,
        "entity_type": "Chemical",
        "canonical_name": "Ammonia",
        "synonyms": [],
        "description": "A colorless alkaline gas. It is formed in the body during decomposition of organic materials during a large number of metabolically important reactions. Note that the aqueous form of ammonia is referred to as AMMONIUM HYDROXIDE.",
        "tree_numbers": [
            "D01.362.075",
            "D01.625.050"
        ]
    },
    ...
]
```
(See `experiments/data/examples/entity_dict.json` for more details.)

### Output

The output is a `networkx.MultiDiGraph` object representing the knowledge graph.

Each node has the following attributes:

- `entity_id` (str): Concept ID (e.g., MeSH ID)
- `entity_type` (str): Type of entity (e.g., Disease, Chemical, Person, Location)
- `name` (str): Canonical name (from Entity Dictionary)
- `description` (str): Textual definition (from Entity Dictionary)
- `doc_key_list` (list[str]): List of document IDs where this entity appears

Each edge has the following attributes:

- `head` (str): Head entity ID
- `tail` (str): Tail entity ID
- `relation` (str): Type of semantic relation
- `doc_key_list` (list[str]): List of document IDs supporting this relation

(See `experiments/data/examples/graph.graphml` for more details.)

### How to Use

```python
from kapipe.graph_construction import GraphConstructor

PATH_TO_DOCUMENTS = "./experiments/data/examples/documents.json"
PATH_TO_TRIPLES = "./experiments/data/examples/additional_triples.json" # Or set to None if unused
PATH_TO_ENTITY_DICT = "./experiments/data/examples/entity_dict.json"

# Initialize the knowledge graph constructor
constructor = GraphConstructor()

# Construct the knowledge graph
graph = constructor.construct_knowledge_graph(
    path_documents_list=[PATH_TO_DOCUMENTS],
    path_additional_triples=PATH_TO_TRIPLES, # Optional
    path_entity_dict=PATH_TO_ENTITY_DICT
)
```
(See `experiments/codes/run_graph_construction.py` for specific examples.)

## 🧱 Community Clustering

### Overview

The **Community Clustering** module partitions the knowledge graph into **semantically coherent subgraphs**, referred to as *communities*.  
Each community represents a localized set of closely related concepts and relations, and serves as a fundamental unit of structured knowledge.

### Input

This module takes as input:

1. Knowledge graph (`networkx.MultiDiGraph`) produced by the **Knowledge Graph Construction** module.

### Output

The output is a list of hierarchical community records (dictionaries), each containing:

- `community_id` (str): Unique ID for the community
- `nodes` (list[str]): List of entity IDs belonging to the community (null for ROOT)
- `level` (int): Depth in the hierarchy (ROOT=-1)
- `parent_community_id` (str): ID of the parent community (null for ROOT)
- `child_community_ids` (list[str]): List of child community IDs (empty for leaf communities)

```json
[
    {
        "community_id": "ROOT",
        "nodes": null,
        "level": -1,
        "parent_community_id": null,
        "child_community_ids": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9"
        ]
    },
    {
        "community_id": "0",
        "nodes": [
            "D016651",
            "D014262",
            "D003866",
            "D003490",
            "D001145"
        ],
        "level": 0,
        "parent_community_id": "ROOT",
        "child_community_ids": [...]
    },
    ...
]
```
(See `experiments/data/examples/communities.json` for more details.)

This hierarchical structure enables multi-level organization of knowledge, particularly useful for coarse-to-fine report generation and scalable retrieval.

### How to Use

```python
from kapipe.community_clustering import (
    HierarchicalLeiden,
    NeighborhoodAggregation,
    TripleLevelFactorization
)

# Initialize the community clusterer
clusterer = HierarchicalLeiden()
# clusterer = NeighborhoodAggregation()
# clusterer = TripleLevelFactorization()

# Apply the community clusterer to the graph
communities = clusterer.cluster_communities(graph)
```
(See `experiments/codes/run_community_clustering.py` for specific examples.)

### Supported Methods

- **Hierarchical Leiden**
    - Recursively applies the Leiden algorithm (Traag et al., 2019) to optimize modularity. Large communities are subdivided until they satisfy a predefined size constraint (default: 10 nodes).
- **Neighborhood Aggregation**
    - Groups each node with its immediate neighbors to form local communities.
- **Triple-level Factorization**
    - Treats each individual (subject, relation, object) triple as an atomic community.

## 📝 Report Generation

### Overview

The **Report Generation** module converts each community into a **natural language report**, making structured knowledge interpretable for both humans and language models.  

### Input

This module takes as input:

1. Knowledge graph (`networkx.MultiDiGraph`) generated by the **Knowledge Graph Construction** module.
1. List of community records generated by the **Community Clustering** module.

### Output

The output is a `.jsonl` file, where each line corresponds to one ***Passage***, a dictionary containing:

- `title` (str): Concice topic summary of the community
- `text` (str): Full natural language description of the community's content

```json
{"title": "Lithium Carbonate and Related Health Conditions", "text": "This report examines the interconnections between Lithium Carbonate, ...."}
{"title": "Phenobarbital and Drug-Induced Dyskinesia", "text": "This report examines the relationship between Phenobarbital, ..."}
{"title": "Ammonia and Valproic Acid in Disorders of Excessive Somnolence", "text": "This report examines the relationship between ammonia and valproic acid, ..."}
...
```
(See `experiments/data/examples/reports.jsonl` for more details.)

✅ The output format is fully compatible with the **Chunking** module, which accepts any dictionary containing a `title` and `text` field.  
Thus, each community report can also be treated as a generic ***Passage***.

### How to Use

```python
from kapipe.report_generation import (
    LLMBasedReportGenerator,
    TemplateBasedReportGenerator
)

PATH_TO_REPORTS = "./experiments/data/examples/reports.jsonl"

# Initialize the report generator
generator = LLMBasedReportGenerator()
# generator = TemplateBasedReportGenerator()
 
# Generate community reports
generator.generate_community_reports(
    graph=graph,
    communities=communities,
    path_output=PATH_TO_REPORTS
)
```
(See `experiments/codes/run_report_generation.py` for specific examples.)

### Supported Methods

- **LLM-based Generation**
    - Uses a large language model (e.g., GPT-4o-mini) prompted with a community content to generate fluent summaries.
- **Template-based Generation**
    - Uses a deterministic format that verbalizes each entity/triple and linearizes them:
        - Entity format: `"{name} | {type} | {definition}"`
        - Triple format: `"{subject} | {relation} | {object}"`

## ✂️ Chunking

### Overview

The **Chunking** module splits each input text into multiple **non-overlapping text chunks**, each constrained by a maximum token length (e.g., 100 tokens).  
This module is essential for preparing context units that are compatible with downstream modules such as retrieval and question answering.  

<!-- It supports any input that conforms to the following format:

- A dictionary containing a `"title"` and `"text"` field.  

This makes the module applicable not only to **community reports**, but also to other types of *Passage* data with similar structure. -->

### Input

This module takes as input:

1. ***Passage***, or a dictionary containing `title` and `text` field.

    - `title` (str): Title of the passage
    - `text` (str): Full natural language description of the passage

```json
{
    "title": "Lithium Carbonate and Related Health Conditions",
    "text": "This report examines the interconnections between Lithium Carbonate, ..."
}
```

### Output

The output is a list of ***Passage*** objects, each containing:
- `title` (str): Same as input
- `text` (str): Chunked portion of the original text, within the specified token window
- Other metadata (e.g., community_id) is carried over

```json
[
    {
        "title": "Lithium Carbonate and Related Health Conditions",
        "text": "This report examines the interconnections between Lithium Carbonate, ..."
    },
    {
        "title": "Lithium Carbonate and Related Health Conditions",
        "text": "This duality necessitates careful monitoring of patients receiving Lithium treatment, ..."
    },
    {
        "title": "Lithium Carbonate and Related Health Conditions",
        "text": "Similarly, cardiac arrhythmias, which involve irregular heartbeats, can pose ..."
    }
    ...
]
```
(See `experiments/data/examples/reports.chunked_w100.jsonl` for more details.)

### How to Use

```python
from kapipe.chunking import Chunker

MODEL_NAME = "en_core_web_sm"  # SpaCy tokenizer
WINDOW_SIZE = 100  # Max number of tokens per chunk

# Initialize the chunker
chunker = Chunker(model_name=MODEL_NAME)

# Chunk the passage
chunked_passages = chunker.split_passage_to_chunked_passages(
    passage=passage,
    window_size=WINDOW_SIZE
)
```
(See `experiments/codes/run_chunking.py` for specific examples.)

## 🔍 Passage Retrieval

### Overview

The **Passage Retrieval** module searches for the top-k most **relevant chunks** given a user query.  
It uses lexical or dense retrievers (e.g., BM25, Contriever) to compute semantic similarity between queries and chunks using embedding-based methods.

### Input

**(1) Indexing**:

During the indexing phase, this module takes as input:

1. List of ***Passage*** objects

**(2) Search**:

During the search phase, this module takes as input:

1. ***Question***, or a dictionary containing:
    - `question_key` (str): Unique identifier for the question
    - `question` (str): Natural language question

```json
{
    "question_key": "question#123",
    "question": "What does lithium carbonate induce?"
}
```
(See `experiments/data/examples/questions.json` for more details.)

### Output

**(1) Indexing**:

The indexing result is automatically saved to the path specified by `index_root` and `index_name`.

**(2) Search**:

The search result for each question is represented as a dictionary containing:
- `question_key` (str): Refers back to the original query
- `contexts` (list[***Passage***]): Top-k retrieved chunks sorted by relevance, each containing:
    - `title` (str): Chunk title
    - `text` (str): Chunk text
    - `score` (float): Similarity score computed by the retriever
    - `rank` (int): Rank of the chunk (1-based)

```json
{
    "question_key": "question#123",
    "contexts": [
        {
            "title": "Lithium Carbonate and Related Health Conditions",
            "text": "This report examines the interconnections between Lithium Carbonate, ...",
            (meta data, if exists)
            "score": 1.5991605520248413,
            "rank": 1
        },
        {
            "title": "Lithium Carbonate and Related Health Conditions",
            "text": "\n\nIn summary, while Lithium Carbonate is an effective treatment for mood disorders, ...",
            (meta data, if exists)
            "score": 1.51018488407135,
            "rank": 2
        },
        ...
    ]
}
```
(See `experiments/data/examples/questions.contexts.json` for more details.)

### How to Use

**(1) Indexing**:

```python
from kapipe.passage_retrieval import Contriever

INDEX_ROOT = "./"
INDEX_NAME = "example"

# Initialize retriever
retriever = Contriever(
    max_passage_length=512,
    pooling_method="average",
    normalize=False,
    gpu_id=0,
    metric="inner-product"
)

# Build index
retriever.make_index(
    passages=passages,
    index_root=INDEX_ROOT,
    index_name=INDEX_NAME
)
```
(See `experiments/codes/run_passage_retrieval.py` for specific examples.)

**(2) Search**:

```python
# Load the index
retriever.load_index(index_root=INDEX_ROOT, index_name=INDEX_NAME)

# Search for top-k contexts
retrieved_passages = retriever.search(queries=[question], top_k=10)[0]
contexts_for_question = {
    "question_key": question["question_key"],
    "contexts": retrieved_passages
}
```

### Supported Methods

- **BM25**
    - A sparse lexical matching model based on term frequency and inverse document frequency.
- **Contriever** (Izacard et al., 2022)
    - A dual-encoder retriever trained with contrastive learning (Izacard et al., 2022). Computes similarity between query and chunk embeddings.
- **ColBERTv2** (Santhanam et al., 2022)
    - A token-level late-interaction retriever for fine-grained semantic matching. Provides higher accuracy with increased inference cost.
    - Note: This method is currently unavailable due to an import error in the external `ragatouille` package ([here](https://github.com/AnswerDotAI/RAGatouille/issues/272)).

## 💬 Question Answering

### Overview

The **Question Answering** module generates an answer for each user query, optionally conditioned on the retrieved context chunks.  
It uses a large language model such as GPT-4o to produce factually grounded and context-aware answers in natural language.

### Input

This module takes as input:

1. ***Question***, or a dictionary containing:
    - `question_key`: Unique identifier for the question
    - `question`: Natural language question string

(See `experiments/data/examples/questions.json` for more details.)

2. A dictionary containing:
    - `question_key`: The same identifier with the ***Question***
    - `contexts`: List of ***Passage*** objects

(See `experiments/data/examples/questions.contexts.json` for more details.)

### Output

The answer is a dictionary containing:

- `question_key` (str): Same as input
- `question` (str): Original question text
- `output_answer` (str): Model-generated natural language answer
- `helpfulness_score` (float): Confidence score generated by the model

```json
{
    "question_key": "question#123",
    "question": "What does lithium carbonate induce?",
    "output_answer": "Lithium carbonate can induce depressive disorder, cyanosis, and cardiac arrhythmias.",
    "helpfulness_score": 1.0
}
```
(See `experiments/data/examples/answers.json` for more details.)

### How to Use

```python
from kapipe.qa import LLMQA
from kapipe import utils

# Initialize the QA module
answerer = LLMQA(path_snapshot="./kapipe/results/qa/llmqa/openai_gpt4o")

# Generate answer
answer = answerer.run(
    question=question,
    contexts_for_question=contexts_for_question
)
```
(See `experiments/codes/run_qa.py` for specific examples.)
