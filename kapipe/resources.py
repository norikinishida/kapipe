from __future__ import annotations

import os

from .datatypes import Config
from . import utils


def resolve_snapshot_path(
    component_name: str,
    method_name: str,
    identifier: str,
) -> str:
    """
    Resolve a public resource identifier to its local snapshot path.

    The root resource configuration maps each identifier to both its method
    name and snapshot path. The method name is checked here to prevent loading
    a snapshot with an incompatible implementation class.
    """

    # KAPipe stores the downloaded-resource configuration under the user's
    # home directory. Keep this path resolution in one place so that method
    # classes do not need to know the global resource layout.
    path_resource_config = os.path.join(
        os.path.expanduser("~"),
        ".kapipe",
        "download",
        "config",
    )

    # Load the complete resource configuration
    root_config: Config = utils.get_hocon_config(
        config_path=path_resource_config
    )
    resource_config: Config = root_config[component_name][identifier]

    # Check that the resource identifier belongs to the expected method
    configured_method_name = resource_config["method"]
    if configured_method_name != method_name:
        raise ValueError(
            f"Identifier '{identifier}' belongs to method "
            f"'{configured_method_name}', not '{method_name}'."
        )

    # Only expose the resolved snapshot path
    path_snapshot: str = resource_config["snapshot"]

    return path_snapshot