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
    "doc_key": "6794356",
    "sentences": [
        "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant .",
        ...
    ],
    "entities": [
        {
            "mention_indices": [0, 3, 7],
            "mention_names": [
                "Tricuspid valve regurgitation",
                "tricuspid regurgitation",
                "tricuspid regurgitation"
            ],
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
        "entity_id": "C009166",
        "canonical_name": "retinol acetate",
        "synonyms": [
            "retinyl acetate",
            "vitamin A acetate"
        ],
        "entity_type": null,
        "description": ""
    },
    {
        "entity_id": "D000641",
        "canonical_name": "Ammonia",
        "synonyms": [],
        "entity_type": "Chemical",
        "description": "A colorless alkaline gas. It is formed in the body during decomposition of organic materials during a large number of metabolically important reactions. Note that the aqueous form of ammonia is referred to as AMMONIUM HYDROXIDE."
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