# Deployment

## Streamlit demo

`streamlit_app.py` loads `demo_assets/cleaned_data.csv` and the three weight files, rebuilds the normalisation from the
first 90% of the data, and predicts on the held-out 10%. Tabs:

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

`python:3.11-slim`, CPU-only `torch==2.5.1` from the PyTorch CPU index (keeps the image small), then
`requirements-demo.txt`. Only `streamlit_app.py`, `models/` and `demo_assets/` are copied. It runs as a non-root user
(uid 1000, as Hugging Face Spaces requires) on port 7860.

## Hugging Face Space

```bash
export HF_TOKEN=hf_xxx                # token with write access
python scripts/deploy_space.py <username>/air-quality-pinn
```

`scripts/deploy_space.py` creates a Docker Space and uploads only what the image needs.

## Updating the demo after retraining

Copy the new `.pth` files from `weights/` into `demo_assets/`. The architecture arguments in `load_everything()`
(`LSTMRegressor(6, 64, 2, 3, 0.2)`, `NeuralODERegressor(6, 64, 64, 3)`, `PINNRegressor(6, 64, 3, 2)`) must match what
you trained.
