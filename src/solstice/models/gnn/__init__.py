# SPDX-License-Identifier: Apache-2.0
"""GNN architectures (require torch + torch_geometric).

gnn_v1:      conditional GNN on the native mesh (solpex-paper baseline)
gnn_encproc: encode-process-decode with latent mesh (gnn_roadmap.md)
"""
from solstice.models.gnn.encproc import GNNEncProcDec  # noqa: F401
from solstice.models.gnn.gnn_v1 import ConditionalGNN  # noqa: F401
