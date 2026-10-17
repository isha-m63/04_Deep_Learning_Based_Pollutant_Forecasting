# Models and results

## Data

UCI Air Quality: 9,357 hourly records (March 2004 – February 2005) from a multisensor device in an Italian city, next
to a reference analyzer.

| Role | Column | Unit |
|---|---|---|
| Target | `co_ground_truth` | mg/m³ |
| Target | `nox_ground_truth` | ppb |
| Target | `no2_ground_truth` | µg/m³ |
| Feature | `sensor_co`, `sensor_nox`, `sensor_no2` | metal-oxide sensor responses |
| Feature | `temperature`, `relative_humidity`, `absolute_humidity` | meteorology |

Cleaning and EDA are in `notebooks/data_cleaning_eda.ipynb`. The cleaned file has 9,357 rows. The first 90%
(8,421 hours) trains, the last 10% (936 hours) tests. With a 3-hour window the test set has 934 samples. Features and
targets are z-scored with training-set mean and standard deviation.

## Models

All use 6 inputs, 64 hidden units and 3 outputs, and train with Adam (lr 1e-3), batch size 32, 50 epochs, MSE loss.

### LSTM
2-layer LSTM, dropout 0.2, zero initial state; input `(batch, 3, 6)`, the last time step's output goes through a linear
layer. This is the only model that uses the full 3-hour window at inference.

### Neural ODE
`z0 = Linear(x_t)`; `dz/dt = MLP(z)` (64 → 64 → 64, ReLU), integrated with `torchdiffeq.odeint` (default solver) over
10 points on `t ∈ [0, 1]`; the final state goes through a linear decoder. "Time" here is a fixed integration interval,
not clock time, so the model treats each hour independently.

### PINN
Tanh MLP, 6 → 64 → 64 → 3 for `num_layers=2` (the first hidden layer plus `num_layers − 1` more). The `dropout` entry in `config.yaml` is not used by this model.
The training loss is

```
loss = MSE(f(x_t), y_t) + lambda_phys · mean( ((f(x_t) − f(x_{t−1})) / Δt)² ),   lambda_phys = 1, Δt = 1
```

The second term penalises hour-to-hour change in the predicted concentrations. It is a smoothness regulariser standing
in for the physics, as the code comment says ("enforce smoothness or zero acceleration as proxy physics constraint").
It is not the diffusion–advection PDE from the project goal. `PINNRegressor.physics_loss` (an autograd residual
`dC/dt − f`) exists but is not called.

## Results

Held-out test set, 934 samples, original units (computed from `demo_assets/` weights and data with the same
normalisation as the app):

| Model | Pollutant | MAE | RMSE | R² |
|---|---|---|---|---|
| LSTM | CO | 0.43 | 0.59 | 0.803 |
| | NOx | 49.1 | 65.5 | 0.870 |
| | NO₂ | 15.8 | 20.5 | 0.802 |
| Neural ODE | CO | 0.37 | 0.54 | 0.836 |
| | NOx | 54.2 | 73.5 | 0.836 |
| | NO₂ | 16.8 | 21.7 | 0.778 |
| PINN | CO | 0.56 | 0.71 | 0.722 |
| | NOx | 76.6 | 91.2 | 0.747 |
| | NO₂ | 18.5 | 23.1 | 0.747 |

Reading the table:
- LSTM and Neural ODE are close; the LSTM wins on NOx and NO₂, the Neural ODE on CO.
- The PINN is weakest on every target. The smoothness penalty does not help here, and a single-hour MLP has less
  context than the LSTM's 3-hour window.
- The saved `evaluation/lstm/lstm_metrics.csv` and `evaluation/neural_ode/neural_ode_metrics.csv` give the same R²
  values (their MAE and RMSE are in normalised units). PINN metrics are now saved in `evaluation/pinn/pinn_metrics.csv`.
- These are one run on one chronological split of a single year, so small differences between models are not
  meaningful.
