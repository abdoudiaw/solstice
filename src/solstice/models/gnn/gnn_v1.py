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
"""gnn_v1 (legacy): conditional GNN on the native mesh (solpex-paper port).

Retired as a released architecture — kept for loading old checkpoints and
as the paper baseline. The released architecture is "gnn" (encproc.py).

Node features (geometry) + FiLM conditioning on control parameters at
every message-passing layer; multi-field per-node output. The baseline
the modernized gnn_encproc must beat.
"""

import torch.nn as nn

from solstice.models.gnn.layers import EdgeConv, NodeFiLM
from solstice.models.registry import register_model


@register_model("gnn_v1")
class ConditionalGNN(nn.Module):
    def __init__(self, node_features=2, param_dim=5, out_features=4,
                 hidden=128, n_layers=6, edge_dim=3, dropout=0.1,
                 film_hidden=128):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            EdgeConv(hidden, hidden, edge_dim) for _ in range(n_layers))
        self.films = nn.ModuleList(
            NodeFiLM(param_dim, hidden, film_hidden) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, out_features),
        )

    def forward(self, x, edge_index, edge_attr, params):
        """x: (N, node_features); params: (N, param_dim) broadcast per node."""
        h = self.node_encoder(x)
        for layer, film in zip(self.layers, self.films):
            h_new = film(layer(h, edge_index, edge_attr), params)
            h = h + self.dropout(h_new)
        return self.decoder(h)
