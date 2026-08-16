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
small target-machine dataset. That matches the REACT transfer-learning
plan.

## Order of work

1. Port current conditional GNN as `gnn_v1` (baseline, reproduces the
   paper).
2. `gnn_encproc`: encoder/decoder mappers + latent mesh + conditional
   layer norm + bounding. Validate against gnn_v1 on the same dataset.
3. Processor variants (graph transformer; flux-aligned edges) as
   config options.
4. FGN-style noise conditioning + CRPS training.
