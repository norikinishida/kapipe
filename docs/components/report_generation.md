# Report Generation (`kapipe.report_generation`)

**Report Generation** converts graph communities into textual reports.

This component takes an entity graph and community records, and returns passages that can be used for retrieval or question answering.

## Input

The input consists of an entity graph, community records, node attribute keys, and edge attribute keys.

| Argument | Type | Description |
|---|---|---|
| `graph` | `networkx.MultiDiGraph` | Entity graph |
| `communities` | `list[dict]` | Community records |
| `node_attr_keys` | `tuple[str, ...]` | Node attributes used to describe nodes |
| `edge_attr_keys` | `tuple[str, ...]` | Edge attributes used to describe relations |

Each community record contains the following fields.

| Field | Type | Description |
|---|---|---|
| `community_id` | `str` | Unique community identifier |
| `nodes` | `list[str] \| None` | Entity IDs belonging to the community |
| `level` | `int` | Depth in the community hierarchy |
| `parent_community_id` | `str \| None` | Parent community ID |
| `child_community_ids` | `list[str]` | Child community IDs |

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

## Output

The output is a list of passages.

Each passage contains generated title and text (with the corresponding community fields).

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Report title |
| `text` | `str` | Report text |

```json
[
    {
        "title": "Lithium Carbonate and Related Health Conditions",
        "text": "This report examines the interconnections between Lithium Carbonate, ....",
        ...
    },
    {
        "title": "Phenobarbital and Drug-Induced Dyskinesia",
        "text": "This report examines the relationship between Phenobarbital, ...",
        ...
    },
    {
        "title": "Ammonia and Valproic Acid in Disorders of Excessive Somnolence",
        "text": "This report examines the relationship between ammonia and valproic acid, ...",
        ...
    },
    ...
]
```

## Supported Methods

| Method | Description |
|---|---|
| Template-based Report Generator | Generates deterministic reports by linearizing nodes and edges |
| LLM-based Report Generator | Generates natural language reports using a proprietary or open-source LLM |

## Usage

### Template-based Report Generation:

```python
from kapipe.report_generation import TemplateBasedReportGenerator

# Instantiate the Report Generation component using a template-based approach
generator = TemplateBasedReportGenerator()

# Generate reports for graph communities
reports = generator.generate_community_reports(
    graph=graph,
    communities=communities,
    node_attr_keys=("name", "entity_type", "description"),
    edge_attr_keys=("relation",),
)
```

### LLM-based Report Generation:

```python
from kapipe.report_generation import LLMBasedReportGenerator

# Instantiate your LLM wrapper
# OpenAI LLM
from kapipe.llms import OpenAILLM
model = OpenAILLM(
    model_name="gpt-5.4-nano",
    max_new_tokens=1024,
)
# HuggingFace LLM
from kapipe.llms import HuggingFaceLLM
model = HuggingFaceLLM(
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    max_new_tokens=1024,
    quantization_bits=4,
)

# Instantiate the Report Generation component using an LLM-based approach
generator = LLMBasedReportGenerator(model=model)

# Generate reports for graph communities
reports = generator.generate_community_reports(
    graph=graph,
    communities=communities,
    node_attr_keys=("name", "entity_type", "description"),
    edge_attr_keys=("relation",),
)
```

## Custom Node and Edge Textualization

You can control how graph content is verbalized by changing `node_attr_keys` and `edge_attr_keys`.

For example, `node_attr_keys=("name", "description")` uses only node names and descriptions.

## Example

See [experiments/report_generation](../../experiments/report_generation) for runnable examples.
