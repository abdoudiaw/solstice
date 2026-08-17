# Examples

Using released models (self-contained bundles, raw physical inputs):

- `predict_state.py` — control parameters -> plasma background
  (`python examples/predict_state.py /path/to/pepc-diiid-state-v1`)
- `predict_sources.py` — plasma state -> EIRENE source terms
  (`python examples/predict_sources.py /path/to/pepc-diiid-sources-v1`)

Training on Colab (need the training store in Drive):

- `train_state.ipynb` — state task (params -> fields incl. q_pol, prad)
- `train_sources.ipynb` — sources task (EIRENE replacement)
