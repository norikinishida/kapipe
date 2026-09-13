import importlib
from typing import Any


__all__ = [
    "AgentTrajectory",
    "AgentStep",
    "Tool",
    "ToolCallingAgent",
]


_NAME_TO_MODULE = {
    "AgentTrajectory": "tool_calling_agent",
    "AgentStep": "tool_calling_agent",
    "Tool": "tool",
    "ToolCallingAgent": "tool_calling_agent",
}


def __getattr__(name: str) -> Any:
    """Function to lazily import public objects."""

    # Validate that the requested name is part of the public API
    if name not in __all__:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'"
        )

    # Import the module that defines the requested public object
    module = importlib.import_module(
        f".{_NAME_TO_MODULE[name]}",
        __name__,
    )

    # Read and cache the requested public object
    value = getattr(module, name)
    globals()[name] = value

    return value