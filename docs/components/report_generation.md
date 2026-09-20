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
| `community_key` | `str` | Unique community identifier |
| `nodes` | `list[str] \| None` | Entity IDs belonging to the community |
| `level` | `int` | Depth in the community hierarchy |
| `parent_community_key` | `str \| None` | Parent community key |
| `child_community_keys` | `list[str]` | Child community keys |

```json
[
    {
        "community_key": "ROOT",
        "nodes": null,
        "level": -1,
        "parent_community_key": null,
        "child_community_keys": [
            "Community(D016651)",
            "Community(D014262)",
            "Community(D002122)",
            ...
        ]
    },
    {
        "community_key": "Community(D016651)",
        "nodes": [
            "D016651",
            "D014262"
        ],
        "level": 0,
        "parent_community_key": "ROOT",
        "child_community_keys": []
    },
    ...
]
```

## Output

The output is a list of passages.

Each passage contains generated title and text (with the corresponding community fields).

| Field | Type | Description |
|---|---|---|
| `passage_key` | `str` | Report passage key derived from the community key |
| `title` | `str` | Report title |
| `text` | `str` | Report text |

```json
[
    {
        "passage_key": "Community(D002122)/report",
        "title": "Calcium Chloride and Cardiac Arrhythmias",
        "text": "This report examines the relationship between Calcium Chloride, a chemical used in medical treatments, and Cardiac Arrhythmias, a significant health condition. The report highlights ...",
        "community_key": "Community(D002122)",
        "nodes": ["D002122", "D001145"],
        "level": 0,
        "parent_community_key": "ROOT",
        "child_community_keys": []
    },
    {
        "passage_key": "Community(D001145)/report",
        "title": "Cardiac Arrhythmias and Inducing Chemicals: Calcium Chloride and Desipramine",
        "text": "This report examines the community surrounding cardiac arrhythmias, focusing on the relationships between the diseases and the chemicals that induce them. The primary entities ...",
        "community_key": "Community(D001145)",
        "nodes": ["D001145", "D002122", "D003891"],
        "level": 0,
        "parent_community_key": "ROOT",
        "child_community_keys": []
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

# Instantiate the template-based Report Generation component
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

# Instantiate the LLM-based Report Generation component
generator = LLMBasedReportGenerator(
    model=model,
    prompt_template_name_or_path="report_generation_01_zeroshot",
)

# Generate reports for graph communities
reports = generator.generate_community_reports(
    graph=graph,
    communities=communities,
    node_attr_keys=("name", "entity_type", "description"),
    edge_attr_keys=("relation",),
)
```

## Custom Prompt Templates

`prompt_template_name_or_path` accepts either the name of a built-in prompt template ([`kapipe/report_generation/prompt_templates/*.txt`](../../kapipe/report_generation/prompt_templates)) or a path to a user-defined prompt template.

A custom prompt template for `LLMBasedReportGenerator` supports the following placeholder.

| Placeholder | Required | Description |
|---|---|---|
| `{content_prompt}` | Yes | Community content constructed from its nodes, relationships, and child-community reports |

The generated community content is inserted into `{content_prompt}` before the prompt is passed to the LLM.

The component validates that `{content_prompt}` is present.

## Example

See [experiments/report_generation](../../experiments/report_generation) for runnable examples.
