# Entity Graph Construction (`kapipe.entity_graph_construction`)

**Entity Graph Construction** builds a directed multi-relational entity graph from relational triples.

This component takes documents with entities and relations, and constructs a `networkx.MultiDiGraph`.

## Input

The input consists of document files, an optional entity dictionary, and optional additional triples.

Documents are provided as a list of JSON file paths.

| Argument | Type | Description |
|---|---|---|
| `documents_path_list` | `list[str] \| None` | Paths to documents with relational triples |
| `entity_dict_path` | `str \| None` | Path to an entity dictionary |
| `additional_triples_path` | `str \| None` | Path to additional triples |

Each document contains the following fields.

| Field | Type | Description |
|---|---|---|
| `doc_key` | `str` | Unique document identifier |
| `entities` | `list[dict]` | Entities in the document |
| `relations` | `list[dict]` | Extracted relations between entities |

Each entity contains the following fields.

| Field | Type | Description |
|---|---|---|
| `mention_indices` | `list[int]` | Indices of mentions belonging to this entity |
| `mention_names` | `list[str]` | Mention strings belonging to this entity |
| `entity_type` | `str` | Entity type |
| `entity_id` | `str` | Concept ID |

Each relation contains the following fields.

| Field | Type | Description |
|---|---|---|
| `arg1` | `int` | Index of the head entity |
| `relation` | `str` | Relation type |
| `arg2` | `int` | Index of the tail entity |

```json
{
    "doc_key": "8800187",
    "sentences": [
        "Effect of calcium chloride and 4 - aminopyridine therapy on desipramine toxicity in rats .",
        ...
    ],
    "mentions": [
        {
            "span": [2, 3],
            "name": "calcium chloride",
            "entity_type": "Chemical",
            "entity_id": "D002122"
        },
        ...
    ],
    "entities": [
        {
            "mention_indices": [0, 11, 16, 22, 26, 27, 30],
            "mention_names": [
                "calcium chloride",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2",
                "CaCl2"
            ],
            "entity_type": "Chemical",
            "entity_id": "D002122"
        },
        ...
    ],
    "relations": [
        {
            "arg1": 0,
            "relation": "CID",
            "arg2": 9
        },
        ...
    ]
}
```

An entity dictionary is represented as a list of entity pages.

| Field | Type | Description |
|---|---|---|
| `entity_id` | `str` | Concept ID |
| `canonical_name` | `str` | Canonical entity name |
| `synonyms` | `list[str]` | Synonyms or aliases |
| `entity_type` | `str` | Entity type |
| `description` | `str` | Entity description |

```JSON
[
    {
        "entity_id": "D000082",
        "canonical_name": "Acetaminophen",
        "entity_type": "Chemical",
        "synonyms": [
            "Hydroxyacetanilide",
            "N-(4-Hydroxyphenyl)acetanilide",
            "Paracetamol",
            "Acetominophen",
            "N-Acetyl-p-aminophenol",
            "p-Acetamidophenol",
            "p-Hydroxyacetanilide",
            "APAP",
            "Acetamidophenol"
        ],
        "description": "Analgesic antipyretic derivative of acetanilide. It has weak anti-inflammatory properties and is used as a common analgesic, but may cause liver, blood cell, and kidney damage."
    },
    {
        "entity_id": "D000409",
        "canonical_name": "Alanine",
        "entity_type": "Chemical",
        "synonyms": [
            "L-Alanine",
            "L Alanine",
            "Alanine, L-Isomer",
            "Alanine, L Isomer",
            "L-Isomer Alanine"
        ],
        "description": "BETA-ALANINE is also available A non-essential amino acid that occurs in high levels in its free state in plasma. It is produced from pyruvate by transamination. It is involved in sugar and acid metabolism, increases IMMUNITY, and provides energy for muscle tissue, BRAIN, and the CENTRAL NERVOUS SYSTEM."
    },
    ...
]
```

Additional triples are represented as a list of triples.

| Field | Type | Description |
|---|---|---|
| `head` | `str` | Head entity ID |
| `relation` | `str` | Relation type |
| `tail` | `str` | Tail entity ID |
| `head_type` | `str` | Head entity type, if available |
| `tail_type` | `str` | Tail entity type, if available |

```json
[
    {
        "head": "D014262",
        "relation": "related_to",
        "tail": "D016651",
        "head_type": "Disease",
        "tail_type": "Disease"
    },
    ...
]
```

## Output

The output is a `networkx.MultiDiGraph`.

Each node represents an entity.

| Attribute | Type | Description |
|---|---|---|
| `entity_id` | `str` | Concept ID |
| `name` | `str` | Canonical entity name |
| `entity_type` | `str` | Entity type |
| `description` | `str` | Entity description |
| `doc_key_list` | `str` | Pipe-separated document IDs supporting the node |

Each edge represents a relation.

| Attribute | Type | Description |
|---|---|---|
| `relation` | `str` | Relation type |
| `doc_key_list` | `str` | Pipe-separated document IDs supporting the edge |

## Supported Methods

| Method | Description |
|---|---|
| Entity Graph Constructor | Constructs a directed multi-relational graph from document-level triples and optional additional triples |

## Usage

### Entity Graph Construction:

```python
from kapipe.entity_graph_construction import EntityGraphConstructor

# Instantiate the Entity Graph Construction component
constructor = EntityGraphConstructor(
    missing_entity_policy="keep",
    missing_entity_description="NO DESCRIPTION.",
)

# Construct an entity graph from extracted triples
graph = constructor.construct_entity_graph(
    documents_path_list=[
        "/path/to/documents_with_triples.json"
    ],
    entity_dict_path="/path/to/entity_dict.json",
    additional_triples_path=None,
)
```

### Entity Graph Construction with Additional Triples:

```python
from kapipe.entity_graph_construction import EntityGraphConstructor

# Instantiate the Entity Graph Construction component
constructor = EntityGraphConstructor(
    missing_entity_policy="keep",
    missing_entity_description="NO DESCRIPTION.",
)

# Construct an entity graph from extracted triples and additional triples
graph = constructor.construct_entity_graph(
    documents_path_list=[
        "/path/to/documents_with_triples.json"
    ],
    entity_dict_path="/path/to/entity_dict.json",
    additional_triples_path="/path/to/additional_triples.json",
)
```

## Custom Entity Dictionaries

You can provide your own entity dictionary through `entity_dict_path`.

The entity dictionary is used to attach canonical names, entity types, and descriptions to graph nodes.

If an entity is missing from the entity dictionary, `missing_entity_policy` controls how the triple is handled.

| Policy | Description |
|---|---|
| `keep` | Keep triples with missing entity pages and use fallback node attributes |
| `drop` | Drop triples whose head or tail entity is missing from the entity dictionary |

## Integration with External Knowledge Graphs

You can provide external triples through `additional_triples_path`.

This is useful when combining extracted triples with an existing knowledge graph or curated relation set.

## Example

See [experiments/entity_graph_construction](../../experiments/entity_graph_construction) for runnable examples.