# GNN modernization roadmap

Where the solpex-paper GNN is, and what state-of-the-art (2025-26)
weather-model practice says to change. References: ECMWF anemoi-core
(PyTorch, `models/encoder_processor_decoder.py`, `layers/processor.py`)
for implementation patterns; Google DeepMind WeatherNext 2 / FGN
(arXiv:2506.10772) for conditioning and ensembles; GraphCast for the
multi-mesh idea.

## Current model (solpex-paper `gnn/conditional_gnn.py`)

- Message passing directly on the native SOLPS mesh, 4-connected edges.
- Control parameters broadcast to every node and concatenated.
- Deterministic; separate hand-rolled ensemble for UQ.

## Target architecture: encode -> process -> decode

1. **Latent (hidden) mesh.** Encoder maps native SOLPS cells to a
   coarser latent mesh in (R, Z); processor runs only on the latent
   mesh; decoder maps back to cells. Only encoder/decoder ever see the
   machine geometry, and they see it through *relative* features
   (offsets, distances, psi differences) — this is what makes weights
   transferable across grids and machines. Anemoi's mapper/processor
   split is the reference implementation.
2. **Swappable processor.** GNN message passing, graph transformer, or
   sparse transformer on the latent mesh — a config choice behind one
   interface (anemoi `layers/processor.py` pattern). Long-range
   attention matters here: SOL transport is strongly anisotropic
   (fast parallel transport along flux surfaces), and 4-connectivity
   propagates information one cell per layer. Additional flux-surface-
   aligned edges (GraphCast multi-mesh analogue) are the cheap version.
3. **Conditioning via conditional layer norm** (FGN
   `norm_conditioning_features`, adaLN/FiLM): control parameters (and
   later machine descriptors) modulate normalization layers instead of
   being concatenated to node features. Cleaner scaling, stronger
   global conditioning.
4. **Output bounding** (anemoi `layers/bounding.py`): enforce
   positivity of ne, Te, Ti and sign/range constraints on sources at
   the output layer instead of hoping the loss learns them.
5. **Calibrated ensembles the FGN way**: a single low-dimensional noise
   vector conditions each forward pass (through the same conditional
   layer norm), trained with CRPS; K independently-seeded members on
   top. Replaces the hand-rolled ensemble; gives control applications
   spread estimates at ~1x inference cost per member.
6. **Node/edge features**: relative geometry only. psi_n, |B|, region
   one-hot, face-set flags at boundaries; per-machine normalization by
   scale lengths (minor radius, connection length) — never absolute
   R, Z.

## Transferability position

Weights shared over nodes/edges + relative features + latent-mesh
indirection means the architecture *admits* cross-geometry transfer
(DIII-D -> KSTAR -> ITER/SPARC). Zero-shot transfer should be treated
as a diagnostic, not a deliverable: regime shift (physics, not
geometry) dominates, so the credible claim is few-shot fine-tuning on a
small target-machine dataset. That matches the DIII-D -> KSTAR transfer-learning
plan.

## Lessons from prior SOLPS surrogate studies

Adopted from the most directly comparable prior work:

- **Predict derived fields directly.** Heat fluxes and radiated power
  are network outputs alongside the state fields (prior SPARC surrogate work
  predicts 195 2D fields), not recomputed from predicted state.
  Scalar QoIs (lambda_q, peak target flux, bolometer integrals) are
  post-processed from predicted fields (`solstice.physics`).
- **Location-dependent, non-linear normalization.** Target-region
  temperatures span ~10 orders of magnitude; per-cell quantile
  transforms or equivalent are required. Plain global
  mean/std normalization is known to fail at the targets.
- **Target-weighted evaluation and loss.** Standard eval report must
  include 1D target profiles (test vs prediction via face sets) and
  target-region metrics, not just full-2D averages; optional loss
  weighting near targets.
- **Target 1D load profiles are learned along the way**, not ignored:
  the deposited heat/particle flux profiles on target face sets are
  auxiliary output heads next to the 2D fields. This is the principled
  version of "weighting near-target locations": direct supervision on
  the quantity that matters (10 MW/m^2 engineering limit; the milestone
  is <5% peak target heat-flux error), at near-zero extra cost since the
  dataset already contains them. No separate target-only models unless
  the joint model demonstrably underperforms a target-only baseline.
- **Physics-consistency metrics.** Power balance of predicted fields
  vs Psep input; particle balance. Reported per released model.
- **Hyperparameter search is not optional** — worst/best spread in a
  search is ~10x error; a large well-tuned MLP
  matches target-only models on full 2D fields, so the MLP stays a
  serious benchmark, not a strawman.
- **Dataset QC as metadata.** Convergence classification per case
  (steady / oscillating / diverged) stored in case attrs and used to
  filter training sets.

## Order of work

1. Port current conditional GNN as `gnn_v1` (baseline, reproduces the
   paper).
2. `gnn_encproc`: encoder/decoder mappers + latent mesh + conditional
   layer norm + bounding. Validate against gnn_v1 on the same dataset.
3. Processor variants (graph transformer; flux-aligned edges) as
   config options.
4. FGN-style noise conditioning + CRPS training.
