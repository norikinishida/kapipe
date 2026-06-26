import importlib
from typing import Any


__all__ = [
    "Answer",
    "CandEntKeyInfo",
    "CandidateEntitiesForDocument",
    "CommunityRecord",
    "Config",
    "ContextsForOneExample",
    "DocKey",
    "Document",
    "Entity",
    "EntityPage",
    "EntityPassage",
    "Mention",
    "Passage",
    "Question",
    "QuestionKey",
    "Triple",
]


_NAME_TO_MODULE = {
    "Answer": "datatypes",
    "CandEntKeyInfo": "datatypes",
    "CandidateEntitiesForDocument": "datatypes",
    "CommunityRecord": "datatypes",
    "Config": "datatypes",
    "ContextsForOneExample": "datatypes",
    "DocKey": "datatypes",
    "Document": "datatypes",
    "Entity": "datatypes",
    "EntityPage": "datatypes",
    "EntityPassage": "datatypes",
    "Mention": "datatypes",
    "Passage": "datatypes",
    "Question": "datatypes",
    "QuestionKey": "datatypes",
    "Triple": "datatypes",
}


def __getattr__(name: str) -> Any:
    """Function to lazily import public objects."""

    # Reject unknown public names immediately
    if name not in __all__:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'"
        )

    # Import the module that defines the requested public object
    module = importlib.import_module(f".{_NAME_TO_MODULE[name]}", __name__)

    # Read the requested public object from the imported module
    value = getattr(module, name)

    # Cache the object to avoid importing the module again for the same name
    globals()[name] = value

    return value