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
"""SOLPS run directory -> canonical case.

All raw-file parsing goes through the ORNL SOLPS-routines package
(solps_routines.readers): b2fgmtry structured/unstructured detection,
guard-cell and index conventions, region labels (cvReg) and face sets.
This module only maps its output onto docs/specs/data_schema.md.

Status: skeleton — wiring to solps_routines pending first dataset build.
"""

from __future__ import annotations


def convert_case(run_dir: str, case_id: str | None = None):
    raise NotImplementedError(
        "pending: map solps_routines.readers output (structured + GOAT wide) "
        "onto the canonical schema"
    )
