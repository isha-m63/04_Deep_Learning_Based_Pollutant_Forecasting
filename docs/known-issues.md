# Known issues and remaining limitations

## Fixed
- Training entry points: `training/train_lstm.py` imports `utils.data_loader` and passes all arguments;
  `train_neural_ode.py` and `train_pinn.py` gained `__main__` blocks. All three read `configs/config.yaml`.
- `config.yaml` now lives in `configs/` with paths relative to the repository root (was absolute Windows paths).
- `utils/data_loader.py` `__main__` loads the config.
- `evaluation/pinn/eval_pinn.py` is implemented; PINN metrics and predictions are saved in `evaluation/pinn/`.
  `eval_lstm.py` and `eval_neural_ode.py` can be run as modules too.
- Normalisation statistics are saved to `weights/normalization.json` (`python -m utils.normalization`) and used by the app.
- The three `.pth` files existed twice (`weights/` and `demo_assets/`); the app now reads `weights/`.
- Tests in `tests/` check that the weights load, that saved R² values reproduce and that every training script runs.
- The README structure was rewritten to match the repository.

## Still open
- Saved metrics are in normalised units; compare with the original-unit table in [models.md](models.md).
- Single split and single run, with no repeated seeds or confidence intervals.
- `data/` is git-ignored except `demo_assets/`, so raw and processed data are not in version control.

## Modelling limitations (design, not bugs)
- **The PINN's physics term is a smoothness penalty**, not the diffusion–advection PDE described in the project goal.
  There is no spatial component, since the data comes from a single location. `models/pinn.py: physics_loss` is unused.
- **The shipped base models are nowcasts.** Targets at hour *t* are predicted from inputs at hour *t* (and *t−1*, *t−2* for
  the LSTM). Day-ahead models (+24/48/72 h) are trained with `--horizon` and are weak: from the last three hours of
  low-cost-sensor readings alone, skill falls to R² ≈ 0.1–0.4 at one day and lower beyond (table in [models.md](models.md)).
  No lagged targets, hour-of-day or weather forecasts are used as inputs.
