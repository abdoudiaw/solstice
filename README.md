# solps-nn

Neural-network surrogates for SOLPS-ITER edge plasmas.

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
import solps_nn

model = solps_nn.zoo.load("diiid-lmode-sources-gnn-v1")
sources = model.predict(plasma_state)
```

See `examples/quickstart.ipynb` (runs on Colab).

## Repository layout

```
src/solps_nn/
  data/        canonical dataset schema + converters (SOLPS output -> canonical)
  graphs/      mesh -> graph construction (edges are derived, never stored)
  models/      registry + architectures (gnn/, unet/, mlp/)
  training/    trainer, losses, normalization (users never need this)
  inference/   checkpoint bundles, ModelInterface — the user-facing API
  coupling/    B2.5/EIRENE socket server, C/Fortran shims
  zoo/         named-weight download
configs/       YAML model/training configs
docs/specs/    data schema, checkpoint bundle, GNN roadmap — the contracts
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
with this repository. Converters in `solps_nn.data.converters` build
canonical cases from SOLPS run directories.

## License

Code: Apache-2.0. Released model weights: CC BY 4.0 (see model cards).
