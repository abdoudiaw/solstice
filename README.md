# SOLSTICE

**S**crape-**O**ff **L**ayer **S**urrogate **T**raining, **I**nference & **C**oupling **E**cosystem — neural-network surrogates for SOLPS-ITER edge plasmas.

Install: `pip install solstice-fusion` (import name: `solstice`).

Two model tasks, one framework:

- **State models** (`params -> plasma`): predict plasma backgrounds
  (Te, Ti, ne, ua, ...) from control parameters. For control-oriented
  digital twins (e.g. coupling to core transport solvers).
- **Source models** (`plasma -> sources`): predict neutral source terms
  (Sp, Sne, Qe, Qi, Sm) from the local plasma state. Drop-in EIRENE
  replacement, callable inline from B2.5.

Any architecture (GNN, U-Net, MLP, graph transformer) registers under a
task and is instantiated from config. Released weights are self-describing
bundles: download a checkpoint and predict anywhere — no training code, no
training data.

## Quick start

```python
from solstice.hub import load_state_bundle, load_source_bundle

state = load_state_bundle("pepc-diiid-state-v1")
fields = state.predict({"pe": 3e6, "pi": 3e6, "core_fueling": 3e20,
                        "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.7})

sources = load_source_bundle("pepc-diiid-sources-v1")
terms = sources.predict(plasma_state, params)
```

Runnable versions: `examples/predict_state.py` and
`examples/predict_sources.py`; training notebooks (Colab):
`examples/train_state.ipynb` and `examples/train_sources.ipynb`.
Requests outside the training parameter box raise a warning
(ensemble-based uncertainty estimates are planned).

## Released models

Names follow `pepc-{machine}-{task}-v{N}` (PEPC: ORNL Power Exhaust and
Particle Control group). Architecture: `gnn` — latent-mesh
encode-process-decode with FiLM conditioning (`mlp_v1` is the
per-field baseline).

## Repository layout

```
src/solstice/
  data/        canonical dataset schema + converters (SOLPS output -> canonical)
  graphs/      mesh -> graph construction (edges are derived, never stored)
  models/      registry + architectures (gnn, mlp baseline)
  inference/   checkpoint loading, profiling
  hub/         released-model bundles: create / load / predict
  physics/     scalar QoIs from predicted fields
configs/       YAML model configs
docs/specs/    data schema, checkpoint bundle, GNN roadmap — the contracts
examples/      predict scripts + Colab training notebooks
```

## Design principles

1. **Unstructured-first mesh.** The canonical mesh is cells/faces/vertices,
   matching GOAT wide-grid SOLPS output. Structured grids are the special
   case where cells carry optional `(ix, iy)` labels. See
   `docs/specs/data_schema.md`.
2. **SOLPS file reading follows
   [SOLPS-routines](https://github.com/ORNL-Fusion)** (Lore/Park) — we do
   not re-derive b2fgmtry/b2fstate parsing or index conventions here.
3. **Self-describing checkpoints.** Weights + config + normalization stats
   + mesh + provenance in one bundle. See `docs/specs/checkpoint_spec.md`.
4. **Graphs are derived.** Datasets store the mesh; loaders build edges
   (face adjacency) at load time, identically for structured and wide grids.

## Data

Training data comes from SOLPS-ITER simulations and is **not** distributed
with this repository. Converters in `solstice.data.converters` build
canonical cases from SOLPS run directories.

## License

Code: Apache-2.0. Released model weights: CC BY 4.0 (see model cards).
