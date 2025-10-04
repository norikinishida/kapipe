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
- [Available Identifiers](#available-identifiers)

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

These components together form an implementation of **retrieval-augmented generation (RAG)** or **graph-based RAG (GraphRAG)**, enabling question answering and reasoning grounded in external (structured) knowledge.

For an example of the GraphRAG pipeline, please see [this example](experiments/notebooks/example.ipynb).

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

If the extraction is successful, you should see a directory `~/.kapipe/download/`, which contains model resources.

## 🧩 Triple Extraction

### Overview

The **Triple Extraction** pipeline identifies relational facts from raw text in the form of (head entity, relation, tail entity) **triples**.

Specifically, this is achieved through the following cascade of components:

1. **Named Entity Recognition (NER):**
    - Detect entity mentions (spans) and classify their types.
1. **Entity Disambiguation Retrieval (ED-Retrieval)**:
    - Retrieve candidate concept IDs from a knowledge base for each mention.
1. **Entity Disambiguation Reranking (ED-Reranking)**:
    - Select the most probable concept ID from the retrieved candidates.
1. **Document-level Relation Extraction (DocRE)**:
    - Extract relational triples based on the disambiguated entity set.

```python
from kapipe.pipelines import TripleExtractionPipeline

# Load the triple extraction pipeline
# Example 1
# (1) NER: GPT-4o-mini with zero-shot NER prompting for user-defined entity types
# (2) ED-Retrieval: Dummy Retriever (assigning the lowercased mention string as the entity ID)
# (3) ED-Reranking: No reranking
# (4) DocRE: GPT-4o-mini with zero-shot DocRE prompting for user-defined relation labels
pipe = TripleExtractionPipeline(
    component_kwargs={
        "ner": {
            "identifier": "gpt4omini_any",
            "entity_types": [{"entity_type": "<entity type>", "definition": "<definition>"}]
        },
        "ed_retrieval": {"identifier": "dummy_entity_retriever"},
        "ed_reranking": {"identifier": "identical_entity_reranker"},
        "docre": {
            "identifier": "gpt4omini_any",
            "relation_labels": [{"relation_label": "<entity type>", "definition": "<definition>"}]
        }
    }
)
# Example 2
# (1) NER: Qwen-2.5-7B-Instruct with few-shot NER prompting for CDR ({Chemical, Disease} entity types)
# (2) ED-Retrieval: BLINK Bi-Encoder trained on CDR and MeSH (2015)
# (3) ED-Reranking: Qwen-2.5-7B-Instruct with few-shot ED-Reranking prompting for CDR and MeSH (2015)
# (4) DocRE: Qwen-2.5-7B-Instruct with few-shot DocRE prompting for CDR (Chemical-Induce-Disease relation label)
pipe = TripleExtractionPipeline(
    component_kwargs={
        "ner": {"identifier": "qwen2_5_7b_cdr", "gpu": 1},
        "ed_retrieval": {"identifier": "blink_bi_encoder_cdr", "gpu": 2},
        "ed_reranking": {"identifier": "qwen2_5_7b_cdr", "gpu": 1},
        "docre": {"identifier": "qwen2_5_7b_cdr", "gpu": 1}
    },
    share_backborn_llm=True
)
# Example 3
# (1) NER: Biaffine-NER trained on CDR ({Chemical, Disease} entity types)
# (2) ED-Retrieval: BLINK Bi-Encoder trained on CDR and MeSH (2015)
# (3) ED-Reranking: BLINK Cross-Encoder trained on CDR and MeSH (2015)
# (4) DocRE: ATLOP trained on CDR (Chemical-Induce-Disease relation label)
pipe = TripleExtractionPipeline(
    component_kwargs={
        "ner": {"identifier": "biaffine_ner_cdr", "gpu": 1},
        "ed_retrieval": {"identifier": "blink_bi_encoder_cdr", "gpu": 1},
        "ed_reranking": {"identifier": "blink_cross_encoder_cdr", "gpu": 2},
        "docre": {"identifier": "atlop_cdr", "gpu": 3}
    }
)

# Apply the triple extraction pipeline to your input document
document = pipe.extract(document)
```
(See `experiments/codes/run_triple_extraction_pipeline.py` for specific examples.)


### Input

This component takes as input:

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
(See `experiments/data/examples/documents.json` for more details.)

Each sub-component takes a ***Document*** object as input, augments it with new fields, and returns it.  
This allows custom metadata to persist throughout the triple extractor.

### Output

The output is also the same-format dictionary (***Document***), augmented with extracted mentions, entities, and triples:

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

<!-- The `identifier` determines the specific models used for each subtask.  
For example, `"biaffinener_blink_blink_atlop_cdr"` uses:

- **NER**: Biaffine-NER (trained on BC5CDR for Chemical and Disease types)
- **ED-Retrieval**: BLINK Bi-Encoder (trained on BC5CDR for MeSH 2015)
- **ED-Reranking**: BLINK Cross-Encoder (trained on BC5CDR for MeSH 2015)
- **DocRE**: ATLOP (trained on BC5CDR for Chemical-Induce-Disease (CID) relation) -->

### Supported Methods

#### Named Entity Recognition (NER)
- **Biaffine-NER** ([`Yu et al., 2020`](https://aclanthology.org/2020.acl-main.577/)): Span-based BERT model using biaffine scoring
- **LLM-NER**: A proprietary/open-source LLM using a NER-specific prompt template

#### Entity Disambiguation Retrieval (ED-Retrieval)
- **Dummy Entity Retriever**: A dummy retriever that assigns the (lowercased) mention string as the entity ID
- **BLINK Bi-Encoder** ([`Wu et al., 2020`](https://aclanthology.org/2020.emnlp-main.519/)): Dense retriever using BERT-based encoders and approximate nearest neighbor search
- **BM25**: Lexical matching
- **Levenshtein**: Edit distance matching

#### Entity Disambiguation Reranking (ED-Reranking)
- **Identical Entity Reranker**: A identical (dummy) reranker that performs no reranking and always chooses the top-1 candidate as the output
- **BLINK Cross-Encoder** (Wu et al., 2020): Reranker using a BERT-based encoder for candidates from the Bi-Encoder
- **LLM-ED**: A proprietary/open-source LLM using a ED-reranking-specific prompt template

#### Document-level Relation Extraction (DocRE)
- **ATLOP** ([`Zhou et al., 2021`](https://ojs.aaai.org/index.php/AAAI/article/view/17717)): BERT-based model for DocRE
- **MA-ATLOP** (Oumaima & Nishida et al., 2024): Mention-agnostic extension of ATLOP
- **MA-QA** (Oumaima & Nishida, 2024): Question-answering style DocRE model
- **LLM-DocRE**: A proprietary/open-source LLM using a DocRE-specific prompt template

👉 A full list of available **identifiers** for each component can be found at the end of this README:  
[Available Identifiers](#available-identifiers)

## 🕸️ Knowledge Graph Construction

### Overview

The **Knowledge Graph Construction** component builds a **directed multi-relational graph** from a set of extracted triples.

- **Nodes** represent unique entities (i.e., concepts).
- **Edges** represent semantic relations between entities.

```python
from kapipe.graph_construction import GraphConstructor

PATH_TO_DOCUMENTS = "./experiments/data/examples/documents.json"
PATH_TO_TRIPLES = "./experiments/data/examples/additional_triples.json" # Or set to None
PATH_TO_ENTITY_DICT = "./experiments/data/examples/entity_dict.json" # Or set to None

# Initialize the knowledge graph constructor
constructor = GraphConstructor()

# Construct the knowledge graph
# Example 1 (use entity dictionary for enriching the node information)
graph = constructor.construct_knowledge_graph(
    path_documents_list=[PATH_TO_DOCUMENTS],
    path_additional_triples=PATH_TO_TRIPLES, # Optional
    path_entity_dict=PATH_TO_ENTITY_DICT
)
# Example 2 (without entity dictionary)
graph = constructor.construct_knowledge_graph(
    path_documents_list=[PATH_TO_DOCUMENTS],
    path_additional_triples=PATH_TO_TRIPLES, # Optional
    path_entity_dict=None
)
```
(See `experiments/codes/run_graph_construction.py` for specific examples.)

### Input

This component takes as input:

1. List of ***Document*** objects with triples, produced by the **Triple Extraction** pipeline

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


3. (optional) ***Entity Dictionary***, or a list of dictionaries, each containing:
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

## 🧱 Community Clustering

### Overview

The **Community Clustering** component partitions the knowledge graph into **semantically coherent subgraphs**, referred to as *communities*.  
Each community represents a localized set of closely related concepts and relations, and serves as a fundamental unit of structured knowledge.

```python
from kapipe.community_clustering import (
    HierarchicalLeiden,
    NeighborhoodAggregation,
    TripleLevelFactorization
)

# Initialize the community clusterer
# Example 1 (Hierarchical Leiden)
clusterer = HierarchicalLeiden()
# Example 2 (Neighborhood Aggregation)
clusterer = NeighborhoodAggregation()
# Example 3 (Triple-Level Factorization)
clusterer = TripleLevelFactorization()

# Apply the community clusterer to the graph
communities = clusterer.cluster_communities(graph)
```
(See `experiments/codes/run_community_clustering.py` for specific examples.)

### Input

This component takes as input:

1. Knowledge graph (`networkx.MultiDiGraph`) produced by the **Knowledge Graph Construction** component.

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

### Supported Methods

- **Hierarchical Leiden**
    - Recursively applies the Leiden algorithm (Traag et al., 2019) to optimize modularity. Large communities are subdivided until they satisfy a predefined size constraint (default: 10 nodes).
- **Neighborhood Aggregation**
    - Groups each node with its immediate neighbors to form local communities.
- **Triple-level Factorization**
    - Treats each individual (subject, relation, object) triple as an atomic community.

## 📝 Report Generation

### Overview

The **Report Generation** component converts each community into a **natural language report**, making structured knowledge interpretable for both humans and language models.  

```python
from kapipe.report_generation import (
    LLMBasedReportGenerator,
    TemplateBasedReportGenerator
)

PATH_TO_REPORTS = "./experiments/data/examples/reports.jsonl"

# Initialize the report generator
# Example 1 (LLM-based; GPT-4o-mini)
generator = LLMBasedReportGenerator(
    llm_backend="openai",
    llm_kwargs={
        "openai_model_name": "gpt-4o-mini",
        "max_new_tokens": 2048,
    }
)
# Example 2 (LLM-based; Qwen-2.5-7B-Instruct)
generator = LLMBasedReportGenerator(
    llm_backend="huggingface",
    llm_kwargs={
        "llm_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "max_new_tokens": 2048,
        "quantization_bits": -1,
    }
)
# Example 3 (Template-based)
generator = TemplateBasedReportGenerator()
 
# Generate community reports
generator.generate_community_reports(
    graph=graph,
    communities=communities,
    path_output=PATH_TO_REPORTS
)
```
(See `experiments/codes/run_report_generation.py` for specific examples.)

### Input

This component takes as input:

1. Knowledge graph (`networkx.MultiDiGraph`) generated by the **Knowledge Graph Construction** component.
1. List of community records generated by the **Community Clustering** component.

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

✅ The output format is fully compatible with the **Chunking** component, which accepts any dictionary containing a `title` and `text` field.  
Thus, each community report can also be treated as a generic ***Passage***.

### Supported Methods

- **LLM-based Generation**
    - Uses a large language model (e.g., GPT-4o-mini) prompted with a community content to generate fluent summaries.
- **Template-based Generation**
    - Uses a deterministic format that verbalizes each entity/triple and linearizes them:
        - Entity format: `"{name} | {type} | {definition}"`
        - Triple format: `"{subject} | {relation} | {object}"`

## ✂️ Chunking

### Overview

The **Chunking** component splits each input text into multiple **non-overlapping text chunks**, each constrained by a maximum token length (e.g., 100 tokens).  
This component is essential for preparing context units that are compatible with downstream components such as retrieval and question answering.  

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

### Input

This component takes as input:

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

## 🔍 Passage Retrieval

### Overview

The **Passage Retrieval** component searches for the top-k most **relevant chunks** given a user query.  
It uses lexical or dense retrievers (e.g., BM25, Contriever) to compute semantic similarity between queries and chunks using embedding-based methods.

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

(See `experiments/codes/run_passage_retrieval.py` for specific examples.)

### Input

**(1) Indexing**:

During the indexing phase, this component takes as input:

1. List of ***Passage*** objects

**(2) Search**:

During the search phase, this component takes as input:

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

The **Question Answering** component generates an answer for each user query, optionally conditioned on the retrieved context chunks.  
It uses a large language model such as GPT-4o to produce factually grounded and context-aware answers in natural language.

```python
from os.path import expanduser
from kapipe.qa import QA
from kapipe import utils

# Initialize the QA component
# Exmaple 1 (without retrieved context)
answerer = QA(identifier="gpt4o_without_context")
# Exmaple 2 (with retrieved context)
answerer = QA(identifier="gpt4o_with_context")

# Generate answer
answer = answerer.answer(
    question=question,
    contexts_for_question=contexts_for_question # or None
)

```
(See `experiments/codes/run_qa.py` for specific examples.)

### Input

This component takes as input:

1. ***Question***, or a dictionary containing:
    - `question_key`: Unique identifier for the question
    - `question`: Natural language question string

(See `experiments/data/examples/questions.json` for more details.)

2. ***Contexts***, or a dictionary containing:
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

## Available Identifiers

The following configurations (`identifier`) are currently available:

| Component | identifier | Method and Domain | Label Set or KB |
| --- | --- | --- | --- |
| NER | `biaffine_ner_linked_docred` | Biaffine-NER trained on Linked-DocRED (Wikipedia articles) | {ORG, LOC, TIME, PER, MISC, NUM} |
| NER | `biaffine_ner_cdr` | Biaffine-NER trained on CDR (biomedical abstracts) | {Chemical, Disease} |
| NER | `gpt4omini_any` | GPT-4o-mini with zero-shot NER prompting | Any (user defined) |
| NER | `gpt4omini_linked_docred` | GPT-4o-mini with few-shot NER prompting for Linked-DocRED | {ORG, LOC, TIME, PER, MISC, NUM} |
| NER | `gpt4omini_cdr` | GPT-4o-mini with few-shot NER prompting for CDR | {Chemical, Disease} |
| NER | `llama3_1_8b_any`, `llama3_1_70b_any`, `qwen2_5_7b_any` | Open-source LLMs with zero-shot NER prompting | Any (user defined) |
| NER | `llama3_1_8b_linked_docred`, `llama3_1_70b_linked_docred`, `qwen2_5_7b_linked_docred` | Open-source LLMs with few-shot NER prompting for Linked-DocRED | {ORG, LOC, TIME, PER, MISC, NUM} |
| NER | `llama3_1_8b_cdr`, `llama3_1_70b_cdr`, `qwen2_5_7b_cdr` | Open-source LLMs with few-shot NER prompting for CDR | {Chemical, Disease} | 
| ED-Retrieval | `dummy_entity_retriever` | Dummy retriever that assigns the mention string as the entity ID | Any |
| ED-Retrieval | `blink_bi_encoder_linked_docred` | BLINK Bi-Encoder model trained on Linked-DocRED | DBpedia (2020.02.01) |
| ED-Retrieval | `blink_bi_encoder_cdr` | BLINK Bi-Encoder model trained on CDR | MeSH (2015) |
| ED-Reranking | `identical_entity_reranker` | Dummy reranker that selects the top-1 candidate as the output without no reranking | Any |
| ED-Reranking | `blink_cross_encoder_linked_docred` | BLINK Cross-Encoder model trained on Linked-DocRED | DBpedia (2020.02.01) |
| ED-Reranking | `blink_cross_encoder_cdr` | BLINK Cross-Encoder model trained on CDR | MeSH (2015) |
| ED-Reranking | `gpt4omini_linked_docred` | GPT-4o-mini with few-shot Entity Disambiguation (reranking) prompting for Linked-DocRED | DBpedia (2020.02.01) |
| ED-Reranking | `gpt4omini_cdr` | GPT-4o-mini with few-shot Entity Disambiguation (reranking) prompting for CDR | MeSH (2015) |
| ED-Reranking | `llama3_1_8b_linked_docred`, `llama3_1_70b_linked_docred`, `qwen2_5_7b_linked_docred` | Open-source LLMs with few-shot Entity Disambiguation (reranking) prompting for Linked-DocRED | DBpedia (2020.02.01) |
| ED-Reranking | `llama3_1_8b_cdr`, `llama3_1_70b_cdr`, `qwen2_5_7b_cdr` | Open-source LLMs with few-shot Entity Disambiguation (reranking) prompting for CDR | MeSH (2015) |
| DocRE | `atlop_linked_docred` | ATLOP trained on Linked-DocRED | 96 labels |
| DocRE | `atlop_cdr` | ATLOP trained on CDR | {Chemical-Induce-Disease} |
| DocRE | `gpt4omini_any` | GPT-4o-mini with few-shot DocRE prompting | Any (user defined) |
| DocRE | `gpt4omini_linked_docred` | GPT-4o-mini with few-shot DocRE prompting for Linked-DocRED | 96 labels |
| DocRE | `gpt4omini_cdr` | GPT-4o-mini with few-shot DocRE prompting for CDR | {Chemical-Induce-Disease} |
| DocRE | `llama3_1_8b_any`, `llama3_1_70b_any`, `qwen2_5_7b_any` | Open-source LLMs with zero-shot DocRE prompting | Any (user defined) |
| DocRE | `llama3_1_8b_linked_docred`, `llama3_1_70b_linked_docred`, `qwen2_5_7b_linked_docred` | Open-source LLMs with few-shot DocRE prompting for Linked-DocRED | 96 labels |
| DocRE | `llama3_1_8b_cdr`, `llama3_1_70b_cdr`, `qwen2_5_7b_cdr` | Open-source LLMs with few-shot DocRE prompting for CDR | {Chemical-Induce-Disease} |
| QA | `gpt4o_without_context` | GPT-4o with QA prompting without retrieved context | Any |
| QA | `gpt4o_with_context` | GPT-4o with QA prompting with retrieved context | Any |

