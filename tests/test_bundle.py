# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from tests.test_import import _tiny_geo


@pytest.fixture()
def tiny_mesh(tmp_path):
    from solstice.data.converters.from_solps import build_structured_mesh
    mesh = build_structured_mesh(_tiny_geo())
    p = tmp_path / "mesh.nc"
    mesh.to_netcdf(p)
    return p, mesh.sizes["cell"]


def test_mlp_bundle_roundtrip(tmp_path, tiny_mesh):
    torch = pytest.importorskip("torch")
    from solstice.hub import create_state_bundle, load_state_bundle
    from solstice.models import build_model

    mesh_path, n_cells = tiny_mesh
    core = build_model("mlp_v1", {"in_dim": 5, "hidden": 8, "out_dim": n_cells})
    pt = {
        "state_dict": core.state_dict(),
        "cell_mean": np.zeros(n_cells), "cell_std": np.ones(n_cells),
        "log10": True,
        "inputs": ["chi", "core_fueling", "dna", "ptot", "puff_D2"],
        "x_mean": np.zeros(5), "x_std": np.ones(5),
    }
    ptp = tmp_path / "diiid-test-state-mlp-te.pt"
    torch.save(pt, ptp)

    out = create_state_bundle(ptp, tmp_path / "bundles", "diiid-test-state-mlp-te-v1",
                              mesh_path, provenance={"dataset": "test"})
    pred = load_state_bundle(out)
    fields = pred.predict({"pe": 1e6, "pi": 1e6, "core_fueling": 1e20,
                           "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.7})
    assert set(fields) == {"te"}
    assert fields["te"].shape == (n_cells,)
    assert np.isfinite(fields["te"]).all() and (fields["te"] >= 0).all()


def test_gnn_bundle_roundtrip(tmp_path, tiny_mesh):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from solstice.hub import create_state_bundle, load_state_bundle
    from solstice.models import build_model

    mesh_path, n_cells = tiny_mesh
    cfg = {"node_features": 2, "param_dim": 5, "out_features": 2,
           "hidden": 8, "n_process_layers": 2}
    core = build_model("gnn_encproc", cfg)
    pt = {
        "state_dict": core.state_dict(), "model_class": "gnn_encproc", "config": cfg,
        "fields": {"te": True, "ua_D1": False},
        "inputs": ["chi", "core_fueling", "dna", "ptot", "puff_D2"],
        "x_mean": np.zeros(5), "x_std": np.ones(5),
        "y_mean": np.zeros((n_cells, 2)), "y_std": np.ones((n_cells, 2)),
        "n_latent": 3,
    }
    ptp = tmp_path / "diiid-test-state-gnn.pt"
    torch.save(pt, ptp)

    out = create_state_bundle(ptp, tmp_path / "bundles", "diiid-test-state-gnn-v1",
                              mesh_path)
    pred = load_state_bundle(out)
    fields = pred.predict({"pe": 1e6, "pi": 1e6, "core_fueling": 1e20,
                           "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.7})
    assert set(fields) == {"te", "ua_D1"}
    assert all(np.isfinite(v).all() for v in fields.values())
