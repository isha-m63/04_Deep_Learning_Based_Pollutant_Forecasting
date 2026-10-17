# AQ Forecasting
**Physics-Informed Neural Networks for Air Quality Forecasting Using Multi-Pollutant and Meteorological Data**

Develop a **physics-based framework** that integrates physical laws governing pollutant transport (e.g., **diffusion–advection PDEs**) with rich **spatiotemporal air quality and meteorological observations** (CO, NO₂, temperature, humidity) from the Kaggle dataset.

---

    S4_AQ_PINN/
    │
    ├── data/
    │ ├── raw/                        # Original Kaggle dataset files
    │ ├── processed/                  # Preprocessed CSV/NetCDF files ready for modeling
    │ └── README.md                   # Guide to data sources and preprocessing steps
    │
    ├── notebooks/
    │ ├── exploratory_data_analysis.ipynb              # EDA on Kaggle data
    │ ├── data_preprocessing.ipynb                     # Data cleaning and normalization
    │ ├── model_training_lstm.ipynb                    # LSTM training notebook
    │ ├── model_training_neural_ode.ipynb              # Neural ODE implementation
    │ └── model_training_pinn.ipynb                    # PINN training with physics incorporation
    │
    ├── models/
    │ ├── lstm.py                      # LSTM model definition
    │ ├── neural_ode.py                # Neural ODE architecture and ODE solvers integration
    │ ├── pinn.py                      # PINN architecture and physics loss implementation
    │ └── utils.py                     # Helper functions for models
    │
    ├── training/
    │ ├── train_lstm.py
    │ ├── train_neural_ode.py
    │ └── train_pinn.py
    │
    ├── evaluation/
    │ └── evaluate_models.py          # Scripts to compare model performance and uncertainty
    │
    ├── configs/
    │ └── config.yaml                 # Hyperparameters and file paths
    │
    └── README.md                     # Detailed project overview, objectives, installation, usage


## Live demo (Streamlit, Docker)

`streamlit_app.py` serves the three trained models (LSTM, Neural ODE, PINN) with model comparison,
a time-series explorer and a what-if sandbox. Weights and the cleaned dataset are bundled in `demo_assets/`.

```bash
docker build -t aq-demo .
docker run -p 7860:7860 aq-demo       # http://localhost:7860
```

Deploy to a Hugging Face Docker Space:

```bash
export HF_TOKEN=hf_xxx
python scripts/deploy_space.py <username>/air-quality-pinn
```

> Note: the models estimate reference-analyzer concentrations from the co-located low-cost sensor
> readings and meteorology of the same hour (soft-sensor / nowcast), not multi-step-ahead forecasts.
