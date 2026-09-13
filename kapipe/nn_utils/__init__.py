import importlib
from typing import Any


__all__ = [
    "AdaptiveThresholdingLoss",
    "Biaffine",
    "FocalLoss",
    "MarginalizedCrossEntropyLoss",
    "get_optimizer",
    "get_optimizer2",
    "get_scheduler",
    "get_scheduler2",
    "make_embedding",
    "make_linear",
    "make_mlp",
    "make_mlp_hidden",
    "make_transformer_encoder",
]


_NAME_TO_MODULE = {
    "AdaptiveThresholdingLoss": "losses",
    "Biaffine": "layers",
    "FocalLoss": "losses",
    "MarginalizedCrossEntropyLoss": "losses",
    "get_optimizer": "optimizers",
    "get_optimizer2": "optimizers",
    "get_scheduler": "schedulers",
    "get_scheduler2": "schedulers",
    "make_embedding": "layers",
    "make_linear": "layers",
    "make_mlp": "layers",
    "make_mlp_hidden": "layers",
    "make_transformer_encoder": "layers",
}


def __getattr__(name: str) -> Any:
    """Function to lazily import public objects."""

    # Validate that the requested name is part of the public API
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