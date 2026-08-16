# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: Apache-2.0
"""Canonical case schema. Contract: docs/specs/data_schema.md.

A Case is an xarray.Dataset with dims (cell, face, vertex) and the
variable names defined in the spec. Helpers here build, validate, and
round-trip cases; they never parse SOLPS files (see data.converters).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

SCHEMA_VERSION = "0"

MESH_CELL_VARS = ("cell_r", "cell_z", "cell_vol", "cell_region")
MESH_FACE_VARS = ("face_vertices", "face_cells", "face_set")
MESH_VERTEX_VARS = ("vx_r", "vx_z")
TOPOLOGIES = ("structured", "wide")


def new_case(mesh: xr.Dataset, fields: dict, sources: dict, inputs: dict,
             attrs: dict) -> xr.Dataset:
    """Assemble a canonical case from a validated mesh and named per-cell arrays."""
    ds = mesh.copy()
    n_cells = ds.sizes["cell"]
    for group, arrays in (("fields", fields), ("sources", sources)):
        for name, values in arrays.items():
            values = np.asarray(values)
            if values.shape[0] != n_cells:
                raise ValueError(f"{group}/{name}: expected {n_cells} cells, got {values.shape}")
            ds[name] = ("cell", values)
    for name, value in inputs.items():
        ds[f"input_{name}"] = float(value)
    ds.attrs.update(attrs)
    ds.attrs["schema_version"] = SCHEMA_VERSION
    validate_case(ds)
    return ds


def validate_case(ds: xr.Dataset) -> None:
    missing = [v for v in MESH_CELL_VARS + MESH_FACE_VARS + MESH_VERTEX_VARS if v not in ds]
    if missing:
        raise ValueError(f"case is missing mesh variables: {missing}")
    topology = ds.attrs.get("topology")
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, got {topology!r}")
    if topology == "structured" and ("cell_ix" not in ds or "cell_iy" not in ds):
        raise ValueError("structured cases must carry cell_ix/cell_iy labels")


def save_case(ds: xr.Dataset, path: str) -> None:
    validate_case(ds)
    if str(path).endswith(".zarr"):
        ds.to_zarr(path, mode="w")
    else:
        ds.to_netcdf(path)


def load_case(path: str) -> xr.Dataset:
    ds = xr.open_zarr(path) if str(path).endswith(".zarr") else xr.open_dataset(path)
    validate_case(ds)
    return ds
