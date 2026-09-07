# Graph Retrieval (`kapipe.graph_retrieval`)

**Graph Retrieval** retrieves a neighborhood subgraph around anchor nodes.

This component indexes a directed graph and returns nodes and directed edges within a specified number of hops from the anchors.

## Input

Indexing takes a `networkx.DiGraph`.

Nodes can represent arbitrary objects.

Node attributes are optional, except when timestamp-based ordering is enabled.

| Attribute | Type | Description |
|---|---|---|
| `timestamp` | `Any` | Value used for node ordering when `use_timestamp=True` |

Each edge represents a directed relation.

| Attribute | Type | Description |
|---|---|---|
| `relation` | `str` | Relation label |
| `explanation` | `str` | Explanation of the relation, if available |

The `relation` attribute is required. Other edge attributes are not included in the output.

Search takes anchor node identifiers and a hop size.

| Argument | Type | Description |
|---|---|---|
| `anchor_node_ids` | `list[str]` | Identifiers of the anchor nodes |
| `hop_size` | `int` | Maximum shortest-path distance from an anchor |

## Output

The output is a tuple containing a node list and an edge list.

Each node preserves the original graph node attributes and adds the following fields.

| Field | Type | Description |
|---|---|---|
| `node_id` | `Any` | Graph node identifier |
| `is_anchor` | `bool` | Whether the node is one of the anchors |

The example below uses a task dependency graph.

Each edge contains the following fields.

| Field | Type | Description |
|---|---|---|
| `head` | `int` | Zero-based index of the head node in the returned node list |
| `relation` | `str` | Relation label |
| `tail` | `int` | Zero-based index of the tail node in the returned node list |
| `explanation` | `str` | Explanation of the relation, if available |

```json
{
    "nodes": [
        {
            "node_id": "task:deploy",
            "name": "Deploy application",
            "timestamp": "2025-03-02",
            "is_anchor": true
        },
        {
            "node_id": "task:test",
            "name": "Run integration tests",
            "timestamp": "2025-03-01",
            "is_anchor": false
        }
    ],
    "edges": [
        {
            "head": 0,
            "relation": "depends_on",
            "tail": 1,
            "explanation": "Deployment depends on successful integration tests."
        }
    ]
}
```

The Python return value is `(nodes, edges)`. The object above shows the two lists together for clarity.

## Supported Methods

| Method | Description |
|---|---|
| Graph Retriever | Retrieves a hop-bounded neighborhood around anchor nodes |

## Usage

### Graph Retrieval:

```python
import networkx as nx

from kapipe.graph_retrieval import GraphRetriever

# Load a directed graph
graph = nx.read_graphml("/path/to/graph.graphml")

# Instantiate the Graph Retrieval component
retriever = GraphRetriever(
    use_timestamp=False,
)

# Build an undirected search index over the directed graph
retriever.make_index(graph=graph)

# Retrieve nodes and directed edges within one hop of the anchors
nodes, edges = retriever.search(
    anchor_node_ids=anchor_node_ids,
    hop_size=1,
)
```

## Neighborhood Retrieval

Neighborhood expansion uses an undirected version of the indexed graph. Incoming and outgoing edges can therefore connect an anchor to a retrieved node.

The returned edges retain their direction from the original graph. All directed edges between retrieved nodes are returned, except self-loops.

With multiple anchors, the component returns the union of their hop-bounded neighborhoods.

Anchors that are absent from the indexed graph are ignored. If no valid anchor reaches the graph, both output lists are empty.

The `hop_size` argument must be greater than or equal to `0`.

## Timestamp Usage

If `use_timestamp` is `True`, every retrieved node must contain a `timestamp` field. Each `timestamp` should be a string in `YYYY-MM-DD` format. The component reorders nodes primarily by timestamp and reorders edges according to the resulting node order. It does not filter nodes or edges by timestamp.

If `use_timestamp` is `False`, timestamps are not required, and the indexed graph's node order is preserved.

## Node Identifier Normalization

Anchor node identifiers are normalized by removing characters that cannot be represented in XML 1.0.
