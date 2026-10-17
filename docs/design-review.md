# Design review

What the original code did against common software-design principles, and what changed. "Open" means not fixed.

| # | Problem found | Principle | Fix |
|---|---|---|---|
| 1 | Three near-identical training loops (`train_lstm`, `train_neural_ode`, `train_pinn`) and three evaluation scripts | **DRY** | `training/trainer.py:fit` and `evaluation/common.py:evaluate`. Verified: retraining with the same seed gives weights and loss curves that differ by exactly 0.0 from the old code (LSTM, Neural ODE, PINN) |
| 2 | Each consumer decided how to build and feed each model (if/elif in the app, hyper-parameters copied into the app and tests) | **Open/Closed**, **DRY** | `models/registry.py`: one `ModelSpec` per model (build from config, forward, weights key, optional regulariser). Adding a model is one registry entry |
| 3 | Absolute Windows paths in `config.yaml`, config in the repo root while scripts expected `configs/`, broken `__main__` blocks | Configuration coupled to one machine; code that cannot run as shipped | `configs/config.yaml` with relative paths, `utils/config.py`, working CLI entry points |
| 4 | Empty placeholder modules (Bayesian Neural ODE, probabilistic LSTM) and the unused `physics_loss` | **YAGNI**, dead code | Removed |
| 5 | The "physics-informed" model is a smoothness penalty, and the project is named *forecasting* while the models predicted the same hour | Names and documentation that overstate what the code does (**honest abstractions**) | Docstrings say what it is; real forecast support added (`--horizon`; day-ahead models at 24/48/72 h; aligned targets, tested). **Open:** the class is still called `PINNRegressor`; a true PDE would need spatial data |
| 6 | Normalisation statistics were recomputed in notebooks, the app and tests | **DRY**, single source of truth | `utils/normalization.py` writes/reads `weights/normalization.json` |
| 7 | Duplicate weight files (`weights/` and `demo_assets/`) | **DRY** | App reads `weights/` |
| 8 | No seeds in the scripts | Reproducibility | `torch.manual_seed(42)` in the shared entry point |
| 9 | A loss file for the PINN sat in the LSTM results folder | Misplaced artifact | Moved to `evaluation/pinn/` |
| 10 | Notebooks still contain their own training code | **DRY** | **Open**, plus stale PINN notebook output |
