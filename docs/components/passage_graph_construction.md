# Passage Graph Construction (`kapipe.passage_graph_construction`)

**Passage Graph Construction** builds a directed graph whose nodes represent passages.

This component takes passages and relation triples, and constructs a `networkx.DiGraph`.

## Input

The input consists of passages and relation triples.

| Argument | Type | Description |
|---|---|---|
| `passages` | `list[dict]` | Passages added as graph nodes |
| `triples` | `list[dict]` | Directed relations between passages |

Each passage must contain a string `passage_key`, which is used as the graph node identifier.

```json
[
    {
        "passage_key": "proposition#001",
        "text": "An independent audit found that the Northbridge payment system processed 99.9% of transactions within two seconds in February 2025.",
        "timestamp": "2025-03-01"
    },
    {
        "passage_key": "proposition#002",
        "text": "A preliminary report found that the Northbridge payment system processed 97.0% of transactions within two seconds in January 2025.",
        "timestamp": "2025-02-01"
    },
    ...
]
```

Each relation triple contains the following fields.

| Field | Type | Description |
|---|---|---|
| `head` | `dict` | Head passage |
| `relation` | `str` | Directed relation from the head passage to the tail passage |
| `tail` | `dict` | Tail passage |
| `explanation` | `str` | Explanation of the relation, if available |

```json
[
    {
        "head": {
            "passage_key": "proposition#001",
            "text": "An independent audit found that the Northbridge payment system processed 99.9% of transactions within two seconds in February 2025.",
            "timestamp": "2025-03-01"
        },
        "relation": "updates",
        "tail": {
            "passage_key": "proposition#002",
            "text": "A preliminary report found that the Northbridge payment system processed 97.0% of transactions within two seconds in January 2025.",
            "timestamp": "2025-02-01"
        },
        "explanation": "The February audit updates the January processing rate from 97.0% to 99.9%."
    },
    ...
]
```

## Output

The output is a `networkx.DiGraph`.

Each node represents a passage. Its identifier is the value of `passage_key`.

All GraphML-compatible scalar fields from the passage are stored as node attributes. `passage_key` and `text` are required passage fields. `timestamp` is an example of optional metadata.

| Attribute | Type | Description |
|---|---|---|
| `passage_key` | `str` | Passage key used as the node identifier |
| `text` | `str` | Passage text |
| `timestamp` | `str` | Passage timestamp, if available |

Fields whose values are `str`, `int`, `float`, or `bool` are preserved. Fields with unsupported values, such as `list`, `dict`, or `None`, are omitted.

Each edge represents a directed relation from a head passage to a tail passage.

| Attribute | Type | Description |
|---|---|---|
| `relation` | `str` | Relation label |
| `explanation` | `str` | Explanation of the relation, if available |

## Supported Methods

| Method | Description |
|---|---|
| Passage Graph Constructor | Constructs a directed passage graph from passages and relation triples |

## Usage

### Passage Graph Construction:

```python
from kapipe.passage_graph_construction import PassageGraphConstructor

# Instantiate the Passage Graph Construction component
constructor = PassageGraphConstructor()

# Construct a directed passage graph
graph = constructor.construct_passage_graph(
    passages=passages,
    triples=triples,
)
```

The resulting graph can be saved in GraphML format with NetworkX.

```python
import networkx as nx

# Save the passage graph in GraphML format
nx.write_graphml(graph, "/path/to/graph.graphml")
```

## Node Construction

All passages in `passages` are added as nodes, including passages that do not occur in any relation triple.

If a head or tail passage in `triples` is absent from `passages`, it is also added as a node.

## Edge Construction

Each relation triple creates a directed edge from its head passage to its tail passage.

Because the output is a `networkx.DiGraph`, it contains at most one edge for each directed node pair. If multiple triples have the same head and tail identifiers, only the first triple is retained.

## GraphML Compatibility

Only `str`, `int`, `float`, and `bool` node and edge attributes are retained. Nested dictionaries, lists, `None`, and other unsupported values are excluded.

Characters that cannot be represented in XML 1.0 are removed from node identifiers and string attributes.

## Example

See [experiments/passage_graph_construction](../../experiments/passage_graph_construction) for runnable examples.
