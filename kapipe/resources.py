from __future__ import annotations

import os

from . import utils


def resolve_snapshot_path(
    component_name: str,
    method_name: str,
    identifier: str,
) -> str:
    """Resolve a public resource identifier to its local snapshot path."""

    # KAPipe stores the downloaded-resource configuration under the user's
    # home directory. Keep this path resolution in one place so that method
    # classes do not need to know the global resource layout.
    resource_config_path = os.path.join(
        os.path.expanduser("~"),
        ".kapipe",
        "download",
        "config",
    )

    # Load the complete resource configuration
    root_config = utils.get_hocon_config(
        config_path=resource_config_path
    )
    resource_config = root_config[component_name][method_name][identifier]

    # Only expose the resolved snapshot path
    snapshot_path: str = resource_config["snapshot"]

    return snapshot_path