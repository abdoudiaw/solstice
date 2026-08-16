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
