"""
Streamlit demo: air-quality pollutant estimation (UCI Air Quality dataset).

Three trained PyTorch models map low-cost metal-oxide sensor signals + meteorology
(sensor_co, sensor_nox, sensor_no2, T, RH, AH) to the reference-analyzer
concentrations of CO, NOx and NO2:
    * LSTM         - uses a 3-hour window
    * Neural ODE   - latent state integrated through a learned vector field
    * PINN         - MLP trained with a temporal-smoothness penalty (not a PDE)
The last 10 % of the series (chronological split) is held out as the test set.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from models.registry import REGISTRY, load_model
from utils.config import load_config
from utils.normalization import FEATURES, TARGETS, load_stats

ROOT = Path(__file__).parent
ASSETS = ROOT / "demo_assets"      # sample data
WEIGHTS = ROOT / "weights"         # trained weights + normalisation statistics
LABELS = {"co_ground_truth": "CO (mg/m³)", "nox_ground_truth": "NOx (ppb)", "no2_ground_truth": "NO₂ (µg/m³)"}
SEQ_LEN = 3
MODELS = [spec.label for spec in REGISTRY.values()]

st.set_page_config(page_title="Air Quality Estimation", page_icon="🌫️", layout="wide")
torch.set_num_threads(1)


HORIZONS = {"Nowcast (same hour)": 0, "Day 1 ahead (+24 h)": 24, "Day 2 ahead (+48 h)": 48, "Day 3 ahead (+72 h)": 72}
START = pd.Timestamp("2004-03-10 18:00")     # the UCI series is hourly from 10 Mar 2004, 18:00
UP, DOWN = "#e5534b", "#2da44e"              # a pollutant that rises is bad (red), one that falls is good (green)


@st.cache_resource
def load_data():
    df = pd.read_csv(ASSETS / "cleaned_data.csv")
    n_train = int(0.9 * len(df))
    # z-score statistics saved by `python -m utils.normalization` (same as notebooks/preprocessing.ipynb)
    stats = load_stats(WEIGHTS / "normalization.json")
    x_mu, x_sd = pd.Series(stats["x_mean"])[FEATURES], pd.Series(stats["x_std"])[FEATURES]
    y_mu, y_sd = pd.Series(stats["y_mean"])[TARGETS], pd.Series(stats["y_std"])[TARGETS]
    return df, n_train, x_mu, x_sd, y_mu, y_sd


@st.cache_resource
def load_nets(horizon: int):
    """label -> (spec, trained module) for every registered model at the given forecast horizon."""
    config = load_config()
    sfx = f"_h{horizon}" if horizon else ""
    return {
        spec.label: (spec, load_model(
            spec, config, WEIGHTS / Path(config["weight_paths"][spec.weights_key]).name.replace(".pth", sfx + ".pth")))
        for spec in REGISTRY.values()
    }


def weights_available(horizon: int) -> bool:
    sfx = f"_h{horizon}" if horizon else ""
    config = load_config()
    return all((WEIGHTS / Path(config["weight_paths"][spec.weights_key]).name.replace(".pth", sfx + ".pth")).exists()
               for spec in REGISTRY.values())


HORIZONS = {label: h for label, h in HORIZONS.items() if weights_available(h)}   # only trained lead times

df, n_train, x_mu, x_sd, y_mu, y_sd = load_data()
task = st.sidebar.selectbox("Task", list(HORIZONS), help="How far ahead the models predict the reference concentrations. "
                            "The nowcast uses the same hour's sensor readings; forecasts use readings up to hour t "
                            "to predict hour t + h.")
HORIZON = HORIZONS[task]
nets = load_nets(HORIZON)


def run_model(nets_: dict, name: str, windows_norm: np.ndarray) -> np.ndarray:
    """windows_norm: (N, SEQ_LEN, 6) normalised features -> (N, 3) in original units."""
    spec, net = nets_[name]
    with torch.no_grad():
        out = spec.forward(net, torch.tensor(windows_norm, dtype=torch.float32))
    return out.numpy() * y_sd.to_numpy() + y_mu.to_numpy()


def make_windows(frame: pd.DataFrame) -> np.ndarray:
    xn = ((frame[FEATURES] - x_mu) / x_sd).to_numpy(dtype="float32")
    return np.stack([xn[i:i + SEQ_LEN] for i in range(len(xn) - SEQ_LEN + 1)])


@st.cache_data
def test_predictions(horizon: int):
    """Held-out predictions of every model `horizon` hours ahead, aligned with their target timestamps."""
    nets_ = load_nets(horizon)
    test = df.iloc[n_train:].reset_index(drop=True)
    w = make_windows(test)[: len(test) - SEQ_LEN + 1 - horizon]
    truth = test[TARGETS].to_numpy()[SEQ_LEN - 1 + horizon:]
    preds = {m: run_model(nets_, m, w) for m in MODELS}
    first_target = n_train + SEQ_LEN - 1 + horizon
    times = pd.DatetimeIndex([START + pd.Timedelta(hours=first_target + i) for i in range(len(truth))])
    return times, truth, preds


def metrics(truth, pred):
    err = pred - truth
    mae = np.abs(err).mean(0)
    rmse = np.sqrt((err ** 2).mean(0))
    r2 = 1 - (err ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)
    return mae, rmse, r2


times, truth, preds = test_predictions(HORIZON)


def fmt_change(v) -> str:
    return "—" if pd.isna(v) else f"{v:+.2f}"


def change_color(v):
    """Red text for any increase, green for any decrease (even +0.01), none for no change."""
    if isinstance(v, str) or pd.isna(v) or round(float(v), 2) == 0:
        return ""
    return f"color: {UP if v > 0 else DOWN}; font-weight: 600"


LEGEND = (f"<span style='color:{UP}'>▲ red = concentration rising (worse air)</span> · "
          f"<span style='color:{DOWN}'>▼ green = falling (better air)</span>")

st.title("🌫️ Air-Quality Pollutant Estimation")
st.caption(f"{task} · UCI Air Quality dataset · LSTM vs Neural ODE vs PINN · "
           "low-cost sensors + meteorology → reference CO / NOx / NO₂")
if HORIZON:
    st.info(f"Each forecast is issued {HORIZON // 24} day(s) before the hour it predicts, from the sensor and weather "
            f"readings of the last 3 hours. Skill falls quickly with lead time (see the table in the first tab).")

tab_compare, tab_series, tab_whatif = st.tabs(["Model comparison", "Time-series explorer", "What-if sandbox"])

# ------------------------------------------------------------ comparison
with tab_compare:
    rows = []
    for m in MODELS:
        mae, rmse, r2 = metrics(truth, preds[m])
        for i, t in enumerate(TARGETS):
            rows.append({"Model": m, "Pollutant": LABELS[t], "MAE": mae[i], "RMSE": rmse[i], "R²": r2[i]})
    res = pd.DataFrame(rows)
    st.subheader(f"Held-out test set ({len(truth)} hourly samples, {len(truth) / 24:.0f} days)")
    st.dataframe(res.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"})
                 .highlight_max(subset=["R²"], color="#cfe8cf"), use_container_width=True, hide_index=True)

    st.subheader("Skill by lead time (R², higher is better)")
    skill = []
    for label, h in HORIZONS.items():
        _, tr, pr = test_predictions(h)
        for m in MODELS:
            r2_ = metrics(tr, pr[m])[2]
            skill.append({"Lead time": label, "Model": m, **{LABELS[t]: r2_[i] for i, t in enumerate(TARGETS)}})
    skill = pd.DataFrame(skill)
    st.dataframe(skill.style.format({LABELS[t]: "{:.2f}" for t in TARGETS})
                 .background_gradient(subset=[LABELS[t] for t in TARGETS], cmap="RdYlGn", vmin=0.0, vmax=0.9),
                 use_container_width=True, hide_index=True)

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
    n_days = len(truth) // 24
    days = c1.slider("Days shown", 1, min(14, n_days), 3)
    first_day = c1.slider("First day", 1, max(n_days - days + 1, 1), 1,
                          help="Day 1 is the first day of the held-out test period.")
    j = TARGETS.index(target)
    sl = slice((first_day - 1) * 24, (first_day - 1 + days) * 24)
    t_win = times[sl]

    fig, ax = plt.subplots(figsize=(10, 4))
    for d in range(days):                                    # alternate-day shading makes each 24 h block visible
        if d % 2 == 0:
            lo = t_win[0].normalize() + pd.Timedelta(days=d)
            ax.axvspan(lo, lo + pd.Timedelta(days=1), color="grey", alpha=0.08, lw=0)
    ax.plot(t_win, truth[sl, j], "k", lw=2, label="Reference analyzer")
    for m in chosen:
        ax.plot(t_win, preds[m][sl, j], lw=1.3, label=m)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.set_xlim(t_win[0], t_win[-1])
    ax.set_xlabel("Day (hour of the predicted value)"); ax.set_ylabel(LABELS[target])
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=0, ha="center")
    c2.pyplot(fig)

    st.markdown("**Daily means and day-over-day change**  \n" + LEGEND, unsafe_allow_html=True)
    series = {"Reference analyzer": truth[sl, j], **{m: preds[m][sl, j] for m in chosen}}
    daily = pd.DataFrame(series, index=t_win).groupby(t_win.normalize()).mean()
    table = pd.DataFrame({"Day": [d.strftime("%a %d %b") for d in daily.index]})
    delta_cols = []
    for col in daily.columns:
        table[col] = daily[col].to_numpy()
        table[f"Δ {col}"] = daily[col].diff().to_numpy()
        delta_cols.append(f"Δ {col}")
    st.dataframe(table.style.format({**{c: "{:.2f}" for c in daily.columns}, **{c: fmt_change for c in delta_cols}}, na_rep="—")
                 .map(change_color, subset=delta_cols), use_container_width=True, hide_index=True)

# ------------------------------------------------------------ what-if
with tab_whatif:
    st.write("Start from a real 3-hour window of the test period, then change the **latest** hour's "
             "readings to see how each model's output shifts.  \n" + LEGEND, unsafe_allow_html=True)
    idx = st.slider("Test window", 0, len(df) - n_train - SEQ_LEN - HORIZON, 100)
    window = df.iloc[n_train + idx: n_train + idx + SEQ_LEN].copy()
    last = window.iloc[-1]
    issued = START + pd.Timedelta(hours=n_train + idx + SEQ_LEN - 1)
    st.caption(f"Window ends {issued:%a %d %b %Y, %H:%M}" +
               (f" · predicting {issued + pd.Timedelta(hours=HORIZON):%a %d %b %Y, %H:%M}" if HORIZON else ""))
    cols = st.columns(3)
    edited = window.copy()
    for k, f in enumerate(FEATURES):
        lo, hi = float(df[f].min()), float(df[f].max())
        edited.iloc[-1, edited.columns.get_loc(f)] = cols[k % 3].slider(
            f, lo, hi, float(last[f]), step=(hi - lo) / 200, key=f"wi_{f}_{idx}")
    w, w0 = make_windows(edited), make_windows(window)

    st.markdown("**Effect of your edits** (output vs. the unedited window)")
    out = st.columns(3)
    for c, m in zip(out, MODELS):
        p, base = run_model(nets, m, w)[0], run_model(nets, m, w0)[0]
        c.markdown(f"**{m}**")
        for i, t in enumerate(TARGETS):
            # numeric delta so that every non-zero change (even +0.01) is coloured; inverse: rising = red
            c.metric(LABELS[t], f"{p[i]:.2f}", delta=round(float(p[i] - base[i]), 2), delta_color="inverse")

    if HORIZON:
        st.markdown(f"**Forecast vs. the latest observation** (reference value at {issued:%H:%M}, window end)")
        out = st.columns(3)
        for c, m in zip(out, MODELS):
            p = run_model(nets, m, w)[0]
            c.markdown(f"**{m}**")
            for i, t in enumerate(TARGETS):
                c.metric(LABELS[t], f"{p[i]:.2f}", delta=round(float(p[i] - last[t]), 2), delta_color="inverse")
    target_row = df.iloc[min(n_train + idx + SEQ_LEN - 1 + HORIZON, len(df) - 1)]
    when = "same hour" if HORIZON == 0 else f"{HORIZON} h after the window"
    st.caption(f"Reference-analyzer values ({when}): " +
               " · ".join(f"{LABELS[t]}: {target_row[t]:.2f}" for t in TARGETS))
