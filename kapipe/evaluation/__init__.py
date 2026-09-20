import importlib
from types import ModuleType


__all__ = [
    "ner",
    "ed",
    "docre",
    "passage_retrieval",
    "qa",
]


def __getattr__(name: str) -> ModuleType:
    """Function to lazily import public subpackages."""

    # Validate that the requested name is part of the public API
    if name not in __all__:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'"
        )

    # Import the requested evaluation subpackage
    module = importlib.import_module(f".{name}", __name__)

    # Cache the subpackage to avoid importing it again for the same name
    globals()[name] = module

    return module