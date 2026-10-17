"""
Streamlit demo: air-quality pollutant estimation (UCI Air Quality dataset).

Three trained PyTorch models map low-cost metal-oxide sensor signals + meteorology
(sensor_co, sensor_nox, sensor_no2, T, RH, AH) to the reference-analyzer
concentrations of CO, NOx and NO2:
    * LSTM         - uses a 3-hour window
    * Neural ODE   - latent state integrated through a learned vector field
    * PINN         - MLP trained with a finite-difference "physics" smoothness term
The last 10 % of the series (chronological split) is held out as the test set.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from models.lstm import LSTMRegressor
from models.neural_ode import NeuralODERegressor
from models.pinn import PINNRegressor
from utils.normalization import load_stats

ROOT = Path(__file__).parent
ASSETS = ROOT / "demo_assets"      # sample data
WEIGHTS = ROOT / "weights"         # trained weights + normalisation statistics
FEATURES = ["sensor_co", "sensor_nox", "sensor_no2", "temperature", "relative_humidity", "absolute_humidity"]
TARGETS = ["co_ground_truth", "nox_ground_truth", "no2_ground_truth"]
LABELS = {"co_ground_truth": "CO (mg/m³)", "nox_ground_truth": "NOx (ppb)", "no2_ground_truth": "NO₂ (µg/m³)"}
SEQ_LEN = 3
MODELS = ["LSTM", "Neural ODE", "PINN"]

st.set_page_config(page_title="Air Quality Estimation", page_icon="🌫️", layout="wide")
torch.set_num_threads(1)


@st.cache_resource
def load_everything():
    df = pd.read_csv(ASSETS / "cleaned_data.csv")
    n_train = int(0.9 * len(df))
    # z-score statistics saved by `python -m utils.normalization` (same as notebooks/preprocessing.ipynb)
    stats = load_stats(WEIGHTS / "normalization.json")
    x_mu, x_sd = pd.Series(stats["x_mean"])[FEATURES], pd.Series(stats["x_std"])[FEATURES]
    y_mu, y_sd = pd.Series(stats["y_mean"])[TARGETS], pd.Series(stats["y_std"])[TARGETS]

    def mk(cls, fname, *a):
        m = cls(*a)
        m.load_state_dict(torch.load(WEIGHTS / fname, map_location="cpu", weights_only=True))
        return m.eval()

    nets = {
        "LSTM": mk(LSTMRegressor, "trained_lstm_model.pth", 6, 64, 2, 3, 0.2),
        "Neural ODE": mk(NeuralODERegressor, "trained_neural_ode_model.pth", 6, 64, 64, 3),
        "PINN": mk(PINNRegressor, "trained_pinn_model.pth", 6, 64, 3, 2),
    }
    return df, n_train, x_mu, x_sd, y_mu, y_sd, nets


df, n_train, x_mu, x_sd, y_mu, y_sd, nets = load_everything()
T_GRID = torch.linspace(0, 1, 10)


def run_model(name: str, windows_norm: np.ndarray) -> np.ndarray:
    """windows_norm: (N, SEQ_LEN, 6) normalised features -> (N, 3) in original units."""
    x = torch.tensor(windows_norm, dtype=torch.float32)
    with torch.no_grad():
        if name == "LSTM":
            out = nets[name](x)
        elif name == "Neural ODE":
            out = nets[name](x[:, -1], T_GRID)
        else:
            out = nets[name](x[:, -1])
    return out.numpy() * y_sd.to_numpy() + y_mu.to_numpy()


def make_windows(frame: pd.DataFrame) -> np.ndarray:
    xn = ((frame[FEATURES] - x_mu) / x_sd).to_numpy(dtype="float32")
    return np.stack([xn[i:i + SEQ_LEN] for i in range(len(xn) - SEQ_LEN + 1)])


@st.cache_data
def test_predictions():
    test = df.iloc[n_train:].reset_index(drop=True)
    w = make_windows(test)
    truth = test[TARGETS].to_numpy()[SEQ_LEN - 1:]
    preds = {m: run_model(m, w) for m in MODELS}
    return test.iloc[SEQ_LEN - 1:].reset_index(drop=True), truth, preds


def metrics(truth, pred):
    err = pred - truth
    mae = np.abs(err).mean(0)
    rmse = np.sqrt((err ** 2).mean(0))
    r2 = 1 - (err ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)
    return mae, rmse, r2


test_df, truth, preds = test_predictions()

st.title("🌫️ Air-Quality Pollutant Estimation")
st.caption("UCI Air Quality dataset · LSTM vs Neural ODE vs PINN · "
           "low-cost sensors + meteorology → reference CO / NOx / NO₂")

tab_compare, tab_series, tab_whatif = st.tabs(["Model comparison", "Time-series explorer", "What-if sandbox"])

# ------------------------------------------------------------ comparison
with tab_compare:
    rows = []
    for m in MODELS:
        mae, rmse, r2 = metrics(truth, preds[m])
        for i, t in enumerate(TARGETS):
            rows.append({"Model": m, "Pollutant": LABELS[t], "MAE": mae[i], "RMSE": rmse[i], "R²": r2[i]})
    res = pd.DataFrame(rows)
    st.subheader(f"Held-out test set ({len(truth)} hourly samples)")
    st.dataframe(res.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"})
                 .highlight_max(subset=["R²"], color="#cfe8cf"), use_container_width=True, hide_index=True)

    pollutant = st.selectbox("Scatter: true vs predicted", TARGETS, format_func=LABELS.get)
    j = TARGETS.index(pollutant)
    cols = st.columns(3)
    for c, m in zip(cols, MODELS):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(truth[:, j], preds[m][:, j], s=6, alpha=0.5)
        lim = [min(truth[:, j].min(), preds[m][:, j].min()), max(truth[:, j].max(), preds[m][:, j].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{m}  (R² = {metrics(truth, preds[m])[2][j]:.3f})")
        ax.set_xlabel("True"); ax.set_ylabel("Predicted"); ax.grid(alpha=0.3)
        c.pyplot(fig)

# ------------------------------------------------------------ time-series
with tab_series:
    c1, c2 = st.columns([1, 4])
    chosen = c1.multiselect("Models", MODELS, default=MODELS)
    target = c1.selectbox("Pollutant", TARGETS, format_func=LABELS.get, key="ts_target")
    span = c1.slider("Hours shown", 24, len(truth), 168, step=24)
    start = c1.slider("Start hour", 0, len(truth) - span, 0, step=12)
    j = TARGETS.index(target)
    sl = slice(start, start + span)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(truth[sl, j], "k", lw=2, label="Reference analyzer")
    for m in chosen:
        ax.plot(preds[m][sl, j], lw=1.3, label=m)
    ax.set_xlabel("Hour (test period)"); ax.set_ylabel(LABELS[target]); ax.legend(); ax.grid(alpha=0.3)
    c2.pyplot(fig)

# ------------------------------------------------------------ what-if
with tab_whatif:
    st.write("Start from a real 3-hour window of the test period, then change the **latest** hour's "
             "readings to see how each model responds.")
    idx = st.slider("Test window", 0, len(test_df) - 1, 100)
    window = df.iloc[n_train + idx: n_train + idx + SEQ_LEN].copy()
    if len(window) < SEQ_LEN:
        window = df.iloc[-SEQ_LEN:].copy()
    last = window.iloc[-1]
    cols = st.columns(3)
    edited = window.copy()
    for k, f in enumerate(FEATURES):
        lo, hi = float(df[f].min()), float(df[f].max())
        edited.iloc[-1, edited.columns.get_loc(f)] = cols[k % 3].slider(
            f, lo, hi, float(last[f]), step=(hi - lo) / 200, key=f"wi_{f}_{idx}")
    w = make_windows(edited)
    out = st.columns(3)
    for c, m in zip(out, MODELS):
        p = run_model(m, w)[0]
        base = run_model(m, make_windows(window))[0]
        c.markdown(f"**{m}**")
        for i, t in enumerate(TARGETS):
            c.metric(LABELS[t], f"{p[i]:.2f}", delta=f"{p[i] - base[i]:+.2f} vs. original window", delta_color="off")
    st.caption("Reference-analyzer values for the original window: " +
               " · ".join(f"{LABELS[t]}: {window.iloc[-1][t]:.2f}" for t in TARGETS))
