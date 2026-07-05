# Community Clustering (`kapipe.community_clustering`)

**Community Clustering** partitions an entity graph into communities (subgraphs).

This component takes a `networkx.MultiDiGraph` and returns hierarchical community records.

## Input

The input is a `networkx.MultiDiGraph`.

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

## Output

The output is a list of community records.

Each community record contains the following fields.

| Field | Type | Description |
|---|---|---|
| `community_id` | `str` | Unique community identifier |
| `nodes` | `list[str] \| None` | Entity IDs belonging to the community |
| `level` | `int` | Depth in the community hierarchy |
| `parent_community_id` | `str \| None` | Parent community ID |
| `child_community_ids` | `list[str]` | Child community IDs |

The `ROOT` community is a virtual root node. Its `nodes` field is `None`.

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

## Supported Methods

| Method | Description |
|---|---|
| Hierarchical Leiden | Detects hierarchical communities using the Leiden algorithm |
| Neighborhood Aggregation | Builds one community for each node and its neighboring nodes |
| Triple-level Factorization | Treats each edge (triple) as one community containing the head and tail nodes |

## Usage

### Hierarchical Leiden:

```python
from kapipe.community_clustering import HierarchicalLeiden

# Instantiate the Community Clustering component using Hierarchical Leiden
clusterer = HierarchicalLeiden(
    max_cluster_size=10,
    use_lcc=True,
)

# Cluster communities in an entity graph
communities = clusterer.cluster_communities(graph=graph)
```

### Neighborhood Aggregation:

```python
from kapipe.community_clustering import NeighborhoodAggregation

# Instantiate the Community Clustering component using Neighborhood Aggregation
clusterer = NeighborhoodAggregation(
    hop_size=1,
)

# Cluster communities in an entity graph
communities = clusterer.cluster_communities(graph=graph)
```

### Triple-level Factorization:

```python
from kapipe.community_clustering import TripleLevelFactorization

# Instantiate the Community Clustering component using Triple-level Factorization
clusterer = TripleLevelFactorization()

# Cluster communities in an entity graph
communities = clusterer.cluster_communities(graph=graph)
```

## Example

See [experiments/community_clustering](../../experiments/community_clustering) for runnable examples.
