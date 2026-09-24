from __future__ import annotations

import os

from huggingface_hub import hf_hub_download, snapshot_download

from . import utils


RESOURCE_REPO_ID = "norikinishida/kapipe-resources"


def resolve_snapshot_path(
    component_name: str,
    method_name: str,
    identifier: str,
) -> str:
    """Download and resolve a public resource identifier."""

    # Define the local directory shared by the resource configuration and snapshots
    download_dir = os.path.join(
        os.path.expanduser("~"),
        ".kapipe",
        "download",
    )

    # Download the latest resource configuration
    resource_config_path = hf_hub_download(
        repo_id=RESOURCE_REPO_ID,
        filename="config",
        local_dir=download_dir,
    )

    # Load the complete resource configuration
    root_config = utils.get_hocon_config(
        config_path=resource_config_path
    )
    resource_config = root_config[component_name][method_name][identifier]

    # Resolve the local snapshot path
    snapshot_path: str = resource_config["snapshot"]

    # Convert the local snapshot path to its repository-relative path
    remote_snapshot_path = os.path.relpath(
        snapshot_path,
        download_dir,
    )

    # Validate that the snapshot is stored under the download directory
    if (
        remote_snapshot_path == os.pardir
        or remote_snapshot_path.startswith(os.pardir + os.sep)
    ):
        raise ValueError(
            f"Snapshot path must be under {download_dir}: {snapshot_path}"
        )

    # Convert the local path separator to the repository path separator
    remote_snapshot_path = remote_snapshot_path.replace(os.sep, "/")

    # Download new or updated files in the requested snapshot
    snapshot_download(
        repo_id=RESOURCE_REPO_ID,
        allow_patterns=f"{remote_snapshot_path}/**",
        local_dir=download_dir,
    )

    # Expose the resolved local snapshot path
    return snapshot_path