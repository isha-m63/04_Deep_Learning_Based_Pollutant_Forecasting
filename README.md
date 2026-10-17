# AQ Forecasting
**Physics-Informed Neural Networks for Air Quality Forecasting Using Multi-Pollutant and Meteorological Data**

Develop a **physics-based framework** that integrates physical laws governing pollutant transport (e.g., **diffusion–advection PDEs**) with rich **spatiotemporal air quality and meteorological observations** (CO, NO₂, temperature, humidity) from the Kaggle dataset.

---

## What is implemented

Three PyTorch models are trained and compared on the **UCI Air Quality dataset** (9,357 hourly records from an
Italian city, low-cost metal-oxide sensors + a co-located reference analyzer):

| Model | Idea | Input |
|---|---|---|
| **LSTM** | 2-layer recurrent baseline | 3-hour window of 6 features |
| **Neural ODE** | Latent state integrated through a learned vector field (`torchdiffeq`) | latest hour |
| **PINN** | MLP with a finite-difference "physics" penalty on consecutive-hour outputs | latest hour (+ previous hour in training) |

Inputs are `sensor_co, sensor_nox, sensor_no2, temperature, relative_humidity, absolute_humidity`; outputs are the
reference **CO, NOx and NO₂** concentrations.

> **Scope note.** The models estimate reference-analyzer concentrations from the co-located low-cost sensor
> readings and meteorology of the same hour (soft-sensor / nowcast), not multi-step-ahead forecasts. The PINN's physics
> term is a smoothness proxy, not the diffusion–advection PDE in the project goal. See
> [docs/models.md](docs/models.md) and [docs/known-issues.md](docs/known-issues.md).

### Held-out results (last 10% of the series, 934 hourly samples, original units)

| Model | R² CO | R² NOx | R² NO₂ | RMSE CO (mg/m³) | RMSE NOx (ppb) | RMSE NO₂ (µg/m³) |
|---|---|---|---|---|---|---|
| LSTM | 0.803 | **0.870** | **0.802** | 0.59 | **65.5** | **20.5** |
| Neural ODE | **0.836** | 0.836 | 0.778 | **0.54** | 73.5 | 21.7 |
| PINN | 0.722 | 0.747 | 0.747 | 0.71 | 91.2 | 23.1 |

## Documentation

| Doc | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Data flow, repo layout, module responsibilities |
| [docs/setup.md](docs/setup.md) | Installation, data, config, training and evaluation |
| [docs/models.md](docs/models.md) | Data and preprocessing, model details, training, full results |
| [docs/deployment.md](docs/deployment.md) | Streamlit demo, Docker, Streamlit Community Cloud |
| [docs/known-issues.md](docs/known-issues.md) | Gaps and bugs found, so you know what not to rely on |

## Project structure
```
├── configs/config.yaml            Hyperparameters and file paths (relative to the repo root)
├── data
│   ├── raw/AirQualityUCI.csv      Original dataset
│   └── processed                  cleaned_data.csv, normalised train/test CSVs, plots
├── demo_assets                    cleaned_data.csv bundled for the demo (data/ is git-ignored)
├── docs                           Project documentation
├── evaluation
│   ├── common.py                  shared evaluation (metrics, predictions, CSV output)
│   ├── lstm/                      eval_lstm.py, metrics, predictions, loss curve, plots
│   ├── neural_ode/                eval_neural_ode.py and its outputs
│   └── pinn/                      eval_pinn.py, metrics, predictions, loss curve
├── models
│   ├── lstm.py                    LSTMRegressor
│   ├── neural_ode.py              ODEFunc, NeuralODERegressor
│   ├── pinn.py                    PINNRegressor
│   └── registry.py                ModelSpec registry: how each model is built and fed (single extension point)
├── notebooks
│   ├── data_cleaning_eda.ipynb    Cleaning + EDA -> cleaned_data.csv
│   ├── preprocessing.ipynb        90/10 split, z-score normalisation
│   └── model_training_{lstm,neural_ode,pinn}.ipynb
├── training
│   ├── trainer.py                 shared training loop + CLI
│   └── train_{lstm,neural_ode,pinn}.py    thin wrappers around trainer.fit
├── tests                          pytest: weights load, saved metrics reproduce, training runs
├── utils                          data_loader.py, config.py, normalization.py
├── weights                        Trained state dicts + normalization.json
├── streamlit_app.py               Interactive demo
├── Dockerfile, requirements.txt, requirements-dev.txt
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

See [docs/setup.md](docs/setup.md) for training and evaluation.

## Live demo (Streamlit, Docker)

`streamlit_app.py` serves the three trained models (LSTM, Neural ODE, PINN) with model comparison,
a time-series explorer with calendar days and day-over-day changes, and a what-if sandbox. Pick `Nowcast` or `Day 1/2/3 ahead` (+24/48/72 h) in the sidebar; rising concentrations are shown in red and falling ones in green. Weights are read from `weights/`, the cleaned dataset from `demo_assets/`.

```bash
docker build -t aq-demo .
docker run -p 7860:7860 aq-demo       # http://localhost:7860
```

**Streamlit Community Cloud:** new app from this repo, branch `main`, file `streamlit_app.py`, Python 3.11.

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## References
- S. De Vito et al., *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring
  scenario*, Sensors and Actuators B 129 (2008) — source of the UCI Air Quality dataset.
- Chen et al. (2018), *Neural Ordinary Differential Equations*, NeurIPS.
- Raissi, Perdikaris & Karniadakis (2019), *Physics-informed neural networks*, J. Comput. Phys. 378.
