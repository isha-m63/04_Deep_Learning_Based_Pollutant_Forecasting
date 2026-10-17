# Architecture

## Data flow

```
data/raw/AirQualityUCI.csv
   │  notebooks/data_cleaning_eda.ipynb
   ▼
data/processed/cleaned_data.csv      9,357 rows: time_index, 3 sensor targets, 3 sensors, T, RH, AH
   │  notebooks/preprocessing.ipynb   chronological 90/10 split, z-score with TRAIN mean/std
   ▼
data/processed/{X,y}_{train,test}_norm.csv
   │  utils/data_loader.py            sliding windows of 3 hours -> (batch, 3, 6), target = last hour
   ▼
training/train_*.py  ->  weights/trained_*_model.pth  +  evaluation/<model>/*_loss_epoch.csv
   │
   ▼
evaluation/eval_*.py  ->  *_metrics.csv, *_predictions.csv, plots
```

`streamlit_app.py` repeats the normalisation from `cleaned_data.csv` (train-split mean and std, `ddof=1`) and loads the
weights from `weights/`.

## Modules

| Path | Responsibility |
|---|---|
| `models/lstm.py` | `LSTMRegressor`: LSTM (batch-first, zero initial state), last time step → linear layer |
| `models/neural_ode.py` | `ODEFunc` (2-layer MLP for dh/dt) and `NeuralODERegressor`: linear encoder → `odeint` over `t ∈ [0, 1]` → linear decoder |
| `models/pinn.py` | `PINNRegressor`: tanh MLP. Contains a `physics_loss` helper (autograd) that the training script does not use |
| `utils/data_loader.py` | `AirQualityDataset` (sliding window, `seq_len=3`) and `get_data_loaders` (no shuffling) |
| `training/train_lstm.py` | Adam + MSE loop; saves state dict and per-epoch losses |
| `training/train_neural_ode.py` | Same loop; uses the last hour of each window and `t = linspace(0, 1, 10)` |
| `training/train_pinn.py` | MSE data loss + `lambda_phys · mean(((f(x_t) − f(x_{t−1})) / Δt)²)` |
| `evaluation/*/eval_*.py` | Compute MAE, RMSE, R² per pollutant, save predictions and plots (LSTM and Neural ODE only) |
| `configs/config.yaml` | Hyperparameters, weight, loss, metric and prediction paths |

## Design decisions

- **Chronological split.** The last 10% of hours is the test set; shuffling a time series would leak neighbouring hours.
  Normalisation statistics come from the training portion only.
- **Same input features for all models.** The three architectures differ only in how they use them, so the comparison is
  about the model, not the data.
- **Data loaders do not shuffle.** Windows stay in time order in both train and test loaders.
- **Targets are normalised.** Metrics in `evaluation/*_metrics.csv` are in z-score units; the Streamlit app and
  [models.md](models.md) report metrics in original units.
