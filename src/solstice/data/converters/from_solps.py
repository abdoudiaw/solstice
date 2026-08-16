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
"""SOLPS run directory -> canonical case (docs/specs/data_schema.md).

All raw-file parsing and index conventions come from the ORNL
SOLPS-routines package (solps_routines.readers): b2fgmtry corners and
neighbor maps (leftix/bottomix handle the poloidal cuts exactly as B2
does), region labels and their semantics (geo["region"]/region_ids),
and balance.nc contents. Nothing is re-derived here.

Structured grids only for now; GOAT wide grids raise until a case is
available to validate against.

Conventions fixed here:
  - interior cells only (guard cells dropped); cell order = C-order
    flatten of [1:nx+1, 1:ny+1], i.e. iy fastest; cell_ix/cell_iy are
    guard-inclusive python indices so any order is reconstructible.
  - b2fgmtry corner order is 0=SW, 1=SE, 2=NW, 3=NE (SW = lower ix,
    lower iy); polygon loop is [0, 1, 3, 2].
  - balance.nc arrays arrive (ny+2, nx+2) [or (ns, ny+2, nx+2)] and are
    transposed to match b2fgmtry's (nx+2, ny+2).
  - te/ti are converted J -> eV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from solstice.data.schema import SCHEMA_VERSION

EV = 1.602176634e-19
CORNER_LOOP = (0, 1, 3, 2)
FACE_SETS = {
    "interior": 0,
    "inner_target": 1,
    "outer_target": 2,
    "core_boundary": 3,
    "pfr_boundary": 4,
    "wall": 5,
}


def _readers():
    try:
        from solps_routines import readers
    except ImportError as err:
        raise ImportError(
            "solps_routines (ORNL SOLPS-routines) is required for SOLPS conversion; "
            "add its src/ to PYTHONPATH"
        ) from err
    return readers


def load_geometry(b2fgmtry_path: str) -> dict:
    geo = _readers().read_b2fgmtry(str(b2fgmtry_path), use_cache=False)
    if geo.get("isUnstructured"):
        raise NotImplementedError(
            "GOAT wide-grid conversion pending a validation case; structured only"
        )
    return geo


def build_structured_mesh(geo: dict) -> xr.Dataset:
    """Canonical mesh from a structured b2fgmtry dict (SOLPS-routines parsed)."""
    nx, ny = int(geo["nx"]), int(geo["ny"])
    crx, cry = np.asarray(geo["crx"]), np.asarray(geo["cry"])
    interior = np.s_[1 : nx + 1, 1 : ny + 1]

    ix2d, iy2d = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing="ij")
    cell_ix, cell_iy = ix2d.reshape(-1), iy2d.reshape(-1)
    n_cells = nx * ny
    corners_r = crx[interior].reshape(n_cells, 4)[:, CORNER_LOOP]
    corners_z = cry[interior].reshape(n_cells, 4)[:, CORNER_LOOP]

    # vertices: dedup corners on rounded coordinates
    pts = np.round(np.stack([corners_r.ravel(), corners_z.ravel()], axis=1), 9)
    uniq, inv = np.unique(pts, axis=0, return_inverse=True)
    cell_corner_vx = inv.reshape(n_cells, 4)  # in CORNER_LOOP order: SW,SE,NE,NW

    # faces from B2's own neighbor maps (cuts handled by leftix/bottomix)
    cell_id = np.full((nx + 2, ny + 2), -1, dtype=np.int64)
    cell_id[interior] = np.arange(n_cells).reshape(nx, ny)
    leftix = np.asarray(geo["leftix_py"], dtype=int)
    leftiy = np.asarray(geo["leftiy_py"], dtype=int)
    botix = np.asarray(geo["bottomix_py"], dtype=int)
    botiy = np.asarray(geo["bottomiy_py"], dtype=int)

    # local vertex ids per side, in CORNER_LOOP order [SW, SE, NE, NW]
    side_vx = {
        "west": (3, 0),   # NW-SW
        "south": (0, 1),  # SW-SE
        "east": (1, 2),   # SE-NE
        "north": (2, 3),  # NE-NW
    }

    face_vertices, face_cells, face_set = [], [], []

    def add_face(c, side, neighbor, boundary_label):
        a, b = side_vx[side]
        face_vertices.append((cell_corner_vx[c, a], cell_corner_vx[c, b]))
        face_cells.append((c, neighbor))
        face_set.append(FACE_SETS[boundary_label if neighbor < 0 else "interior"])

    region_vol = np.asarray(geo["region"])[:, :, 0]  # region_ids['vol'] semantics
    is_core_row = region_vol == geo["region_ids"]["vol"]["is_core"]
    # geometric inner/outer target identification (inner = smaller R)
    mean_r_west = crx[1, 1 : ny + 1, :].mean()
    mean_r_east = crx[nx, 1 : ny + 1, :].mean()
    west_label = "inner_target" if mean_r_west < mean_r_east else "outer_target"
    east_label = "outer_target" if west_label == "inner_target" else "inner_target"

    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            c = cell_id[ix, iy]
            ln = cell_id[leftix[ix, iy], leftiy[ix, iy]]
            add_face(c, "west", ln, west_label if ix == 1 else "interior")
            bn = cell_id[botix[ix, iy], botiy[ix, iy]]
            add_face(
                c, "south", bn,
                ("core_boundary" if is_core_row[ix, iy] else "pfr_boundary")
                if iy == 1 else "interior",
            )
            if ix == nx:
                add_face(c, "east", -1, east_label)
            if iy == ny:
                add_face(c, "north", -1, "wall")

    # interior faces were added once from each side; dedup on the cell pair
    fv = np.asarray(face_vertices)
    fc = np.asarray(face_cells)
    fs = np.asarray(face_set)
    pair_key = np.where(
        fc[:, 1] >= 0,
        np.minimum(fc[:, 0], fc[:, 1]) * (nx * ny + 1) + np.maximum(fc[:, 0], fc[:, 1]),
        -np.arange(1, len(fc) + 1),  # boundary faces are unique
    )
    _, keep = np.unique(pair_key, return_index=True)
    keep.sort()
    fv, fc, fs = fv[keep], fc[keep], fs[keep]

    bb = np.asarray(geo["bb"])[interior].reshape(n_cells, 4)
    ds = xr.Dataset(
        {
            "cell_r": ("cell", corners_r.mean(axis=1)),
            "cell_z": ("cell", corners_z.mean(axis=1)),
            "cell_corners_r": (("cell", "corner"), corners_r),
            "cell_corners_z": (("cell", "corner"), corners_z),
            "cell_vol": ("cell", np.asarray(geo["vol"])[interior].reshape(-1)),
            "cell_b": (("cell", "bcomp"), bb),
            "cell_region": ("cell", region_vol[interior].reshape(-1).astype(np.int8)),
            "cell_region_x": ("cell", np.asarray(geo["region"])[:, :, 1][interior].reshape(-1).astype(np.int8)),
            "cell_region_y": ("cell", np.asarray(geo["region"])[:, :, 2][interior].reshape(-1).astype(np.int8)),
            "cell_ix": ("cell", cell_ix),
            "cell_iy": ("cell", cell_iy),
            "face_vertices": (("face", "two"), fv),
            "face_cells": (("face", "two"), fc),
            "face_set": ("face", fs.astype(np.int8)),
            "vx_r": ("vertex", uniq[:, 0]),
            "vx_z": ("vertex", uniq[:, 1]),
        },
        attrs={
            "topology": "structured",
            "nx": nx,
            "ny": ny,
            "region_ids": json.dumps(geo["region_ids"]),
            "face_sets": json.dumps(FACE_SETS),
            "schema_version": SCHEMA_VERSION,
        },
    )
    return ds


def _species_tags(species_chars: np.ndarray) -> list[str]:
    tags = []
    for row in np.asarray(species_chars):
        s = "".join(c.decode() if isinstance(c, bytes) else str(c) for c in row).strip()
        tags.append(s.replace("+", "").replace(" ", ""))
    return tags


def read_case_fields(run_dir: str, nx: int, ny: int) -> tuple[dict, list[str]]:
    """Per-cell named field arrays (interior, flattened) from balance.nc."""
    bal = _readers().read_balance(str(Path(run_dir) / "balance.nc"))
    if bal is None:
        raise FileNotFoundError(f"no balance.nc in {run_dir}")

    def cells(a2d):
        return np.asarray(a2d).T[1 : nx + 1, 1 : ny + 1].reshape(-1)

    tags = _species_tags(bal["species"])
    fields = {
        "te": cells(bal["te"]) / EV,
        "ti": cells(bal["ti"]) / EV,
        "ne": cells(bal["ne"]),
    }
    for i, tag in enumerate(tags):
        fields[f"na_{tag}"] = cells(bal["na"][i])
        fields[f"ua_{tag}"] = cells(bal["ua"][i])
    return fields, tags


def read_case_inputs(run_dir: str) -> tuple[dict, dict]:
    """Scalar control parameters and status metadata from params.json."""
    meta = json.loads((Path(run_dir) / "params.json").read_text())
    p = meta.get("inputs", {})
    core = p.get("core", {})
    inputs = {
        "pe_core": p.get("power", {}).get("Pe_W"),
        "pi_core": p.get("power", {}).get("Pi_W"),
        # canonical name is core_fueling (see data_schema.md). Legacy
        # params.json mislabels it "core.density_m-3"; fixed metadata is
        # expected to use "core.fueling" — both are accepted here.
        "core_fueling": core.get("fueling", core.get("density_m-3")),
    }
    for gas, rec in p.get("gas_puffing", {}).get("targets", {}).items():
        inputs[f"puff_{gas}"] = rec.get("value")
    for key in ("dna", "hci", "hce"):
        if key in p.get("transport", {}):
            inputs[key] = p["transport"][key].get("value")
    inputs = {k: float(v) for k, v in inputs.items() if v is not None}
    return inputs, meta


def convert_case(run_dir: str, mesh: xr.Dataset) -> xr.Dataset:
    """One structured SOLPS run -> canonical case on a prebuilt mesh."""
    nx, ny = int(mesh.attrs["nx"]), int(mesh.attrs["ny"])
    fields, tags = read_case_fields(run_dir, nx, ny)
    inputs, meta = read_case_inputs(run_dir)

    ds = mesh.copy()
    for name, values in fields.items():
        ds[name] = ("cell", values)
    for name, value in inputs.items():
        ds[f"input_{name}"] = value
    ds.attrs.update(
        machine=meta.get("machine", "unknown"),
        case_id=meta.get("case", {}).get("case_id", Path(run_dir).name),
        species=" ".join(tags),
        convergence_status="unclassified",
    )
    return ds
