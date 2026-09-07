# Context Formatting (`kapipe.context_formatting`)

**Context Formatting** converts input data into text that can be passed to an LLM.

The input format and conversion behavior depend on the selected method.

## Input

The input depends on the selected method.

## Output

The output is a string.

## Supported Methods

| Method | Description |
|---|---|
| Graph Verbalizer | Verbalizes proposition nodes and their directed relations as structured text |

## Usage

### Graph Verbalization:

```python
from kapipe.context_formatting import GraphVerbalizer

# Instantiate the Graph Verbalizer
formatter = GraphVerbalizer(
    use_timestamp=True,
)

# Convert proposition nodes and edges into LLM-readable text
text = formatter.convert(
    nodes=nodes,
    edges=edges,
)
```

## Graph Verbalizer

### TODO:

The current Graph Verbalizer is tightly coupled to the proposition representation and relation semantics of [ProStruct-RAG](../pipelines/prostruct_rag_pipeline.md). These assumptions should be separated from the general Context Formatting interface.

### Node Attributes:

The Graph Verbalizer uses the following node attributes.

| Attribute | Type | Required | Description |
|---|---|---|---|
| `text` | `str` | Yes | Proposition text |
| `timestamp` | `str` | Only when `use_timestamp=True` | Proposition timestamp |

Other node attributes are ignored.

### Node References:

Nodes are assigned one-based labels such as `P1`, `P2`, and `P3` according to their order in the input node list.

Edge endpoints are zero-based indices into that same list. The Graph Verbalizer does not reorder nodes.

For each node, both outgoing and incoming relations are verbalized.

### Relation Formatting:

The following relation labels have specialized verbalizations.

| Relation | Outgoing Form | Incoming Form |
|---|---|---|
| `updates` | `updates` | `is updated by` |
| `contradicts` | `contradicts` | `is contradicted by` |
| `supports` | `supports` | `is supported by` |

These three labels are matched case-insensitively and are verbalized in the order shown above.

Other relation labels are preserved and verbalized using the generic phrase `has the "<relation>" relation to`. Unknown labels follow their first occurrence order in the edge list.

### Timestamp Usage:

If `use_timestamp` is `True`, every node must contain a `timestamp` field. The timestamp is displayed below each proposition label and beside referenced propositions. Relation text refers to outgoing targets as earlier propositions and incoming sources as later propositions. The Graph Verbalizer does not sort nodes, compare timestamps, or validate temporal order. The input must already follow the intended temporal interpretation.

If `use_timestamp` is `False`, timestamps are not required or displayed. Relation text uses the temporally neutral term `propositions`.

## Example

See [experiments/context_formatting](../../experiments/context_formatting) for runnable examples.
