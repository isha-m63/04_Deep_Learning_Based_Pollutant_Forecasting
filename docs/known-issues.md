# Known issues and unfinished work

Found while documenting the code. Nothing below has been changed.

## Does not run as written
- `training/train_lstm.py` `__main__` imports `data.load_data` (no such module; the loader is `utils.data_loader`) and
  calls `train_lstm(...)` without the required `hidden_size`, `num_layers` and `dropout`.
- `utils/data_loader.py` `__main__` uses an undefined `config`.
- `config.yaml` contains only absolute Windows paths and sits in the repo root, while code and notebooks expect
  `configs/config.yaml` or `C:/02_AtmPhy_ML/S4_AQ_PINN/config.yaml`.
- `evaluation/pinn/eval_pinn.py`, `models/bayesian_neural_ode.py`, `models/probabilistic_lstm.py`,
  `training/train_bayesian_neural_ode.py` and `training/train_probabilistic_lstm.py` are empty files.

## Modeling
- **The PINN's physics term is a smoothness penalty**, not the diffusion–advection PDE described in the project goal.
  There is no spatial component, since the data comes from a single location.
- **These are nowcasts, not forecasts.** Targets at hour *t* are predicted from inputs at hour *t* (and *t−1*, *t−2* for
  the LSTM).
- `models/pinn.py: physics_loss` is unused, and the autograd derivative it takes would need a time input that the
  model does not have.
- `notebooks/model_training_pinn.ipynb` has stale output: it prints a 2-output network and LSTM metrics. The saved
  weights in `weights/` and `demo_assets/` load correctly with 3 outputs.
- No PINN metrics or predictions are saved in `evaluation/`; `evaluation/lstm/pinn_loss_epoch.csv` is in the LSTM
  folder.
- Saved metrics are in normalised units; compare with the original-unit table in [models.md](models.md).
- Normalisation parameters are not saved to disk. The demo recomputes them from the first 90% of `cleaned_data.csv`.
- Single split and single run, with no repeated seeds or confidence intervals.

## Documentation drift
- The earlier README listed files that do not exist (`data/README.md`, `models/utils.py`, `evaluation/evaluate_models.py`,
  `configs/config.yaml`, `notebooks/exploratory_data_analysis.ipynb`, `data_preprocessing.ipynb`) under a `S4_AQ_PINN/`
  root. The structure in the README now reflects the actual repository.
- `data/` is git-ignored except `demo_assets/`, so raw and processed data are not in version control.
