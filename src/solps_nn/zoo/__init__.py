# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: Apache-2.0
"""Named released weights. Names follow the checkpoint spec:
{machine}-{regime}-{task}-{arch}[-mini]-v{N}."""

from __future__ import annotations

# name -> huggingface repo id (populated at first release)
RELEASES: dict[str, str] = {}


def load(name: str):
    """Download a released bundle by name and return a ready ModelInterface."""
    from solps_nn.inference import load_checkpoint

    try:
        repo_id = RELEASES[name]
    except KeyError:
        raise KeyError(f"unknown release {name!r}; available: {sorted(RELEASES)}") from None
    from huggingface_hub import snapshot_download

    return load_checkpoint(snapshot_download(repo_id))
