# =========================================================================================
# (C) (or copyright) 2026. UT-Battelle, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract DE-AC05-00OR22725 with
# UT-Battelle, LLC, which manages Oak Ridge National Laboratory (ORNL) for the U.S.
# Department of Energy (DOE). The U.S. Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
# to reproduce, prepare derivative works, distribute copies to the public, perform
# publicly and display publicly, and to permit others to do so. The DOE will provide
# public access to these results in accordance with the DOE Public Access Plan
# (http://energy.gov/downloads/doe-public-access-plan).
# =========================================================================================
# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: Apache-2.0
"""Named released weights, torch.hub-style. Names follow the checkpoint spec:
{machine}-{regime}-{task}-{arch}[-mini]-v{N}."""

from __future__ import annotations

# name -> huggingface repo id (populated at first release)
RELEASES: dict[str, str] = {}


def load(name: str):
    """Download a released bundle by name and return a ready ModelInterface."""
    from solstice.inference import load_checkpoint

    try:
        repo_id = RELEASES[name]
    except KeyError:
        raise KeyError(f"unknown release {name!r}; available: {sorted(RELEASES)}") from None
    from huggingface_hub import snapshot_download

    return load_checkpoint(snapshot_download(repo_id))
