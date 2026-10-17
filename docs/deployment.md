# Deployment

## Streamlit demo

`streamlit_app.py` loads `demo_assets/cleaned_data.csv`, the three weight files and `weights/normalization.json`, and predicts on the held-out 10%. Tabs:

- **Model comparison.** Metrics table (MAE, RMSE, R² per model and pollutant) and true-vs-predicted scatter plots.
- **Time-series explorer.** Reference analyzer vs. model predictions over a chosen window of the test period.
- **What-if sandbox.** Take a real 3-hour test window, change the latest hour's sensor and meteorology values with
  sliders, and see how each model's output shifts.

Run locally with `streamlit run streamlit_app.py`.

## Docker

```bash
docker build -t aq-demo .
docker run -p 7860:7860 aq-demo       # http://localhost:7860
```

`python:3.11-slim`, CPU-only PyTorch (the `--extra-index-url` line in `requirements.txt` keeps the image small).
Only `streamlit_app.py`, `models/`, `utils/`, `weights/` and `demo_assets/` are copied. It runs as a non-root user
on port 7860.

## Streamlit Community Cloud

New app from this repository: branch `main`, main file `streamlit_app.py`, Python 3.11. It installs `requirements.txt`
(including the CPU PyTorch index) and needs no training step.

## Updating the demo after retraining

The app reads `weights/` directly, so retraining updates it automatically (re-run `python -m utils.normalization` if
the data changed). The architecture arguments in `load_everything()`
(`LSTMRegressor(6, 64, 2, 3, 0.2)`, `NeuralODERegressor(6, 64, 64, 3)`, `PINNRegressor(6, 64, 3, 2)`) must match what
you trained.
