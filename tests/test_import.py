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
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import xarray as xr

import solps_nn
from solps_nn.graphs import cell_adjacency_edges
from solps_nn.models import register_model, build_model


def test_registry_roundtrip():
    @register_model("dummy")
    class Dummy:
        def __init__(self, k=1):
            self.k = k

    assert build_model("dummy", {"k": 3}).k == 3
    with pytest.raises(KeyError):
        build_model("nope", {})


def test_cell_adjacency_edges():
    # two cells sharing one face; one boundary face
    case = xr.Dataset(
        {
            "face_cells": (("face", "two"), np.array([[0, 1], [1, -1]])),
            "cell_r": ("cell", np.array([0.0, 1.0])),
            "cell_z": ("cell", np.array([0.0, 0.0])),
        }
    )
    edge_index, edge_attr = cell_adjacency_edges(case)
    assert edge_index.shape == (2, 2)  # undirected -> both directions
    assert np.allclose(edge_attr[:, 2], 1.0)
