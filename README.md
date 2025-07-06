# KAPipe: A Modular Pipeline for Knowledge Acquisition

## Table of Contents

- [🤖 KAPipe Overview](#kapipe-overview)
- [📦 Installation](#-installation)
- [🧩 Triple Extraction](#-triple-extraction)
- [🕸️ Knowledge Graph Construction](#-knowledge-graph-construction)
- [🧱 Community Clustering](#-community-clustering)
- [📝 Report Generation](#-report-generation)
- [✂️ Chunking](#-chunking)
- [🔍 Passage Retrieval](#-passage-retrieval)
- [💬 Question Answering](#-question-answering)

## 🤖 KAPipe Overview

**KAPipe** is a modular pipeline for comprehensive **knowledge acquisition** from unstructured documents.  
It supports extraction, organization, retrieval, and utilization of knowledge, serving as a core framework for building intelligent systems that reason over structured knowledge.
These components together form a powerful implementation of **graph-based retrieval-augmented generation (GraphRAG)**, enabling question answering and reasoning grounded in structured knowledge.

KAPipe provides the following functionalities:

- 🧩**Triple Extraction**  
    - Extract relational facts in the form of (head entity, relation, tail entity) from raw text using BERT-based models or proprietary/open-source LLMs.

- 🕸️**Knowledge Graph Construction**  
    - Build a symbolic knowledge graph from extracted triples, optionally augmented with external ontologies or knowledge bases (e.g., UMLS).

- 🧱**Community Clustering**  
    - Cluster the knowledge graph into semantically coherent subgraphs (*communities*).

- 📝**Report Generation**  
    - Generate textual summaries of graph communities to support interpretable retrieval and reasoning.

- ✂️**Chunking**  
    - Split passages into fixed-size chunks based on a predefined token length (e.g., n=300).

- 🔍**Passage Retrieval**  
    - Retrieve relevant chunks using lexical or dense retrieval to support downstream tasks.

- 💬**Question Answering**  
    - Answer questions using retrieved chunks as context.

KAPipe is designed to be modular, composable, and domain-adaptable, making it suitable for various applications such as biomedical text mining, scientific discovery, and explainable AI.

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

The **Triple Extraction** module identifies relational facts from raw text in the form of (head entity, relation, tail entity) triples.

This is achieved through the following cascade of subtasks:

1. **Named Entity Recognition (NER):**
    - Detect entity mentions (spans) and classify their types.
1. **Entity Disambiguation Retrieval (ED-Retrieval)**:
    - Retrieve candidate concept IDs from a knowledge base for each mention.
1. **Entity Disambiguation Reranking (ED-Reranking)**:
    - Select the most probable concept ID from the retrieved candidates.
1. **Document-level Relation Extraction (DocRE)**:
    - Extract relational triples based on the disambiguated entity set.

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

**Document-level Relation Extraction (DocRE)**
- **ATLOP** ([`Zhou et al., 2021`](https://ojs.aaai.org/index.php/AAAI/article/view/17717)): BERT-based model for DocRE
- **MA-ATLOP** (Oumaima & Nishida et al., 2024): Mention-agnostic extension of ATLOP
- **MA-QA** (Oumaima & Nishida, 2024): Question-answering style DocRE model
- **LLM-DocRE**: A proprietary/open-source LLM using a DocRE-specific prompt template and few-shot examples

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

Each subtask takes a ***Document*** object as input, augments it with new fields, and returns it. This allows custom metadata to persist throughout the pipeline.

`.jsonl`ファイル (各行は`"title"`と`"text"`を持つ辞書; 後述の ***Passage*** ) から、list of ***Document*** objectsを作成する方法を提供しています。

```python
from kapipe.chunkers import Chunker
from kapipe import utils

SPACY_MODEL_NAME=en_core_sci_md
PATH_TO_PASSAGES = "./raw_passages.jsonl"
PATH_TO_DOCUMENTS = "./documents.json

# Initialize the Chunker
chunker = Chunker(model_name=SPACY_MODEL_NAME)

documents = []
with open(PATH_TO_PASSAGES) as f:
    for line in f:
        # Load Passage
        passage = json.loads(line.strip())
        # Convert Passage to Document
        document = chunker.convert_passage_to_document(
            doc_key=f"Passage#{len(documents)}",
            passage=passage,
            do_tokenize=True
        )
        documents.append(document)

# Save the Documents
utils.write_json(PATH_TO_DOCUMENTS, documents)
```

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

### How to Use

```python
import kapipe.triple_extraction

IDENTIFIER = "biaffinener_blink_blink_atlop_cdr"

# Load the Triple Extraction pipeline
pipe = kapipe.extractors.load(
    identifier=IDENTIFIER,
    gpu_map={
        "ner": 0,
        "ed_retrieval": 0,
        "ed_reranking": 2,
        "docre": 3
    }
)

# Apply the pipeline to your input document
document = pipe(document)
```

The `identifier` determines the specific models used for each subtask.  
For example, `"biaffinener_blink_blink_atlop_cdr"` uses:

- **NER**: Biaffine-NER (trained on BC5CDR for Chemical and Disease types)
- **ED-Retrieval**: BLINK Bi-Encoder (trained on BC5CDR for MeSH 2015)
- **ED-Reranking**: BLINK Cross-Encoder (trained on BC5CDR for MeSH 2015)
- **DocRE**: ATLOP (trained on BC5CDR for Chemical-Induce-Disease (CID) relation)

## 🕸️ Knowledge Graph Construction

### Overview

The **Knowledge Graph Construction** module builds a directed multi-relational graph from a set of extracted triples.

- **Nodes** represent unique entities (i.e., concepts).
- **Edges** represent semantic relations between entities.

### Input

This module takes as input:

1. List of ***Document*** objects with triples

1. ***Entity Dictionary***, or a list of dictionaries, each containing:
    - `entity_id` (str): Unique concept ID
    - `canonical_name` (str): Official name of the concept
    - `entity_type` (str): Type/category of the concept
    - `synonyms` (list[str]): A list of alternative names
    - `description` (str): Textual definition of the concept
```JSON
[
    {
        "entity_id": "D000001",
        "entity_index": 0,
        "entity_type": "Chemical",
        "canonical_name": "Calcimycin",
        "synonyms": [],
        "description": "An ionophorous, polyether antibiotic from Streptomyces chartreusensis. It binds and transports CALCIUM and other divalent cations across membranes and uncouples oxidative phosphorylation while inhibiting ATPase of rat liver mitochondria. The substance is used mostly as a biochemical tool to study the role of divalent cations in various biological systems."
    },
    {
        "entity_id": "D000002",
        "entity_index": 1,
        "entity_type": "Chemical",
        "canonical_name": "Temefos",
        "synonyms": [
            "Temephos"
        ],
        "description": "for use to kill or control insects, use no qualifiers on the insecticide or the insect; appropriate qualifiers may be used when other aspects of the insecticide are discussed such as the effect on a physiologic process or behavioral aspect of the insect; for poisoning, coordinate with ORGANOPHOSPHATE POISONING An organothiophosphate insecticide."
    },
    ...
]
```

3. (optional) ***Additional Triples*** (existing KBs), or a list of dictionaries, each containing:
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

### How to Use

```python
from kapipe.graph_construction import GraphConstructor

PATH_TO_DOCUMENTS = "./documents.json"
PATH_TO_ENTITY_DICT = "./entity_dict.json"
PATH_TO_TRIPLES = "./triples.json"  # Or set to None if unused

# Initialize the knowledge graph constructor
constructor = GraphConstructor()

# Construct the knowledge graph
graph = constructor.construct_knowledge_graph(
    path_documents_list=[PATH_TO_DOCUMENTS],
    path_additional_triples=PATH_TO_TRIPLES  # Optional
    path_entity_dict=PATH_TO_ENTITY_DICT,
)
```

## 🧱 Community Clustering

### Overview

The **Community Clustering** module partitions the knowledge graph into **semantically coherent subgraphs**, referred to as *communities*.  
Each community represents a localized set of closely related concepts and relations, and serves as a fundamental unit for downstream tasks such as report generation and retrieval.

### Supported Methods

- **Hierarchical Leiden**
    - Recursively applies the Leiden algorithm (Traag et al., 2019) to optimize modularity. Large communities are subdivided until they satisfy a predefined size constraint (default: 10 nodes).
- **Neighborhood Aggregation**
    - Groups each node with its immediate neighbors to form local communities.
- **Triple-level Factorization**
    - Treats each individual (subject, relation, object) triple as an atomic community.

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
        "child_community_ids": ["<community ID>", ...]
    },
    {
        "community_id": "<community id>",
        "nodes": ["<entity ID>", ...],
        "level": 0,
        "parent_community_id": "<parent community id>",
        "child_community_ids": ["<child community id>", ...]
    },
    ...
]
```

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

## 📝 Report Generation

### Overview

The **Report Generation** module converts each community into a **natural language summary**, making structured knowledge interpretable for both humans and language models.  

### Supported Methods

- **LLM-based Generation**
    - Uses a large language model (e.g., GPT-4o-mini) prompted with a community content to generate fluent summaries.
- **Template-based Generation**
    - Uses a deterministic format that verbalizes each entity/triple and linearizes them:
        - Entity format: `"{name} | {type} | {definition}"`
        - Triple format: `"{subject} | {relation} | {object}"`

### Input

This module takes as input:

1. Knowledge graph (`networkx.MultiDiGraph`) generated by the **Knowledge Graph Construction** module.
1. List of community records created by the **Community Clustering** module

### Output

The output is a `.jsonl` file, where each line corresponds to one ***Passage***, a dictionary containing:

- `title` (str): Concice topic summary of the community
- `text` (str): Full natural language description of the community's content

```json
{
    "title": "Methyldopa and Its Associated Health Risks",
    "text": "The community surrounding Methyldopa consists of various diseases that are chemically induced by this antihypertensive agent. Methyldopa is linked to a range of adverse health effects, including hypotension, ...",
}
```

✅ The output format is fully compatible with the Chunking module,  
which accepts any dictionary containing a `title` and `text` field.  
Thus, each community report can also be treated as a generic *Passage*.

### How to Use

```python
from kapipe.graphs import (
    generate_community_reports_by_llm,
    generate_community_reports_by_template,
)

PATH_TO_REPORTS = "./reports.jsonl"

# Generate reports using an LLM
generate_community_reports_by_llm(
    graph=graph,
    communities=communities,
    path_output=PATH_TO_REPORTS
)
# Alternatively, use template-based generation
generate_community_reports_by_template(
    graph=graph,
    communities=communities,
    path_output=PATH_TO_REPORTS
)
```

## ✂️ Chunking

### Overview

The **Chunking** module splits each input text into multiple **non-overlapping text chunks**, each constrained by a maximum token length (e.g., 100 tokens).  
This module is essential for preparing context units that are compatible with downstream modules such as retrieval and question answering.  

<!-- It supports any input that conforms to the following format:

- A dictionary containing a `"title"` and `"text"` field.  

This makes the module applicable not only to **community reports**, but also to other types of *Passage* data with similar structure. -->

### Input

This module takes as input:

1. ***Passage***, or a dictionary containing a `"title"` and `"text"` field.

### Output

The output is a list of ***Passage*** objects, each containing:
- `title` (str): Same as input
- `text` (str): Chunked portion of the original text, within the specified token window
- Other metadata (e.g., community_id) is carried over

```json
[
    {
        "title": "Methyldopa and Its Associated Health Risks",
        "text": "The community surrounding Methyldopa consists of ..."
    },
    {
        "title": "Methyldopa and Its Associated Health Risks",
        "text": "The drug's impact on liver health is particularly ..."
    },
    ...
]
```

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

## 🔍 Passage Retrieval

### Overview

The **Passage Retrieval** module searches for the top-k most relevant **chunks** given a user query (question).  
It uses dense retrievers (e.g., Contriever, ColBERTv2) to compute semantic similarity between queries and chunks using embedding-based methods.

This module is critical for selecting informative contexts before passing them to the Question Answering (QA) stage.

### Supported Methods

- **Contriever** (Izacard et al., 2022)
    - A dual-encoder retriever trained with contrastive learning (Izacard et al., 2022). Computes similarity between query and chunk embeddings.
- **ColBERTv2** (Santhanam et al., 2022)
    - A token-level late-interaction retriever for fine-grained semantic matching. Provides higher accuracy with increased inference cost.

### Input

During the indexing phase, this module takes as input:

- List of ***Passage*** objects

During the search phase, this module takes as input:

- ***Question***, or a dictionary containing:
    - `question_key` (str): Unique identifier for the question
    - `question` (str): Natural language question

```json
{
    "question_key": "Q123",
    "question": "What cardiac conditions are associated with lithium exposure in newborns?"
}
```

### Output

The search result for each question is represented as a dictionarty containing:
- `question_key` (str): Refers back to the original query
- `contexts` (list[***Passage***]): Top-k retrieved chunks (***Passages***) sorted by relevance, each containing:
    - `title` (str): Same as the source chunk
    - `text` (str): Same as the source chunk
    - `score` (float): Similarity score computed by the retriever
    - `rank` (int): Rank of the chunk (1-based)

```json
{
    "question_key": "Q123",
    "contexts": [
        {
          "title": "Lithium-induced cardiac disorders in newborns",
          "text": "This community discusses the effects of lithium exposure...",
          "score": 1.7323,
          "rank": 1
        },
        ...
    ]
}
```

### How to Use

**(1) Indexing**:

```python
from kapipe.passage_retrieval import (
    Contriever,
    ColBERTv2Retriever
)

INDEX_ROOT = "./"
INDEX_NAME = "community_chunks"

# Initialize retriever
retriever = Contriever(
    max_passage_length=512,
    pooling_method="average",
    normalize=False,
    gpu_id=0,
    metric="inner-product"
)
# retriever = ColBERTv2Retriever()

# Build index
retriever.make_index(
    passages=passages,
    index_root=INDEX_ROOT,
    index_name=INDEX_NAME
)
```

**(2) Search**:

```python
from kapipe.retrievers import Contriever, ColBERTv2Retriever

# Initalize the retriever, and load the index
retriever = ...
retriever.load_index(index_root=INDEX_ROOT, index_name=INDEX_NAME)

# Search for top-k contexts
retrieved_passages = retriever.search(queries=[question], top_k=10)[0]
contexts_for_question = {
    "question_key": question["question_key"],
    "contexts": retrieved_passages
}
```

## 💬 Question Answering

### Overview

The **Question Answering** module generates an answer for each user query, optionally conditioned on the retrieved context chunks.  
It uses a large language model such as GPT-4o to produce factually grounded and context-aware answers in natural language.

### Input

This module takes as input:

1. ***Question***, or a dictionary containing:
    - `question_key`: Unique identifier for the question
    - `question`: Natural language question string

```json
{
    "question_key": "Q123",
    "question": "What cardiac conditions are associated with lithium exposure in newborns?"
}
```

2. ***Contexts***, or a dictionary containing:
    - `question_key`: The same identifier with the ***Question***
    - `contexts`: List of ***Passage*** objects

```json
{
    "question_key": "Q123",
    "contexts": [
        {
          "title": "Lithium-induced cardiac disorders in newborns",
          "text": "This community discusses the effects of lithium exposure...",
          "score": 1.7323,
          "rank": 1
        },
        ...
    ]
}
```

### Output

The answer is a dictionary containing:

- `question_key` (str): Same as input
- `question` (str): Original question text
- `output_answer` (str): Model-generated natural language answer

```json
{
    "question_key": "Q123",
    "question": "What cardiac conditions are associated with lithium exposure in newborns?",
    "output_answer": "Lithium exposure during pregnancy is associated with tricuspid regurgitation, atrial flutter, and cardiac arrhythmia in newborns.",
}
```

### How to Use

```python
from kapipe.qa import LLMQA
from kapipe import utils

# Load the LLM configuration
config = utils.get_hocon_config(
    config_path="./kapipe/results/qa/llmqa/openai_gpt4o/config"
)
answerer = LLMQA(config=config)

# Generate answer
answer = answerer.run(
    question=question,
    contexts_for_question=contexts_for_question
)
```
