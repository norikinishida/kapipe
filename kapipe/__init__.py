import importlib
from types import ModuleType


__version__ = "0.3.0"


__all__ = [
    # Core
    "datatypes",
    "utils",

    # Components for Knowledge Extraction
    "ner",
    "ed_retrieval",
    "ed_reranking",
    "docre",
    "proposition_extraction",
    "proposition_relation_extraction",
    "proposition_relation_refinement",

    # Components for Knowledge Organization
    "entity_graph_construction",
    "passage_graph_construction",
    "community_clustering",
    "report_generation",
    "chunking",

    # Components for Knowledge Retrieval
    "passage_retrieval",
    "graph_retrieval",

    # Components for Knowledge Utilization
    "context_formatting",
    "qa",

    # Others
    "llms",
    "nn_utils",
    "evaluation",

    # Pipelines
    "pipelines",

    # Agents
    "agents",
]


def __getattr__(name: str) -> ModuleType:
    """Function to lazily import public subpackages."""

    # Validate that the requested name is part of the public API
    if name not in __all__:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'"
        )

    # Import the requested subpackage or module only when it is requested
    module = importlib.import_module(f".{name}", __name__)

    # Cache the imported object to avoid importing it again for the same name
    globals()[name] = module

    return module
