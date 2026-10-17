"""Weights load, saved metrics reproduce, and every training entry point runs for one epoch."""
import numpy as np
import pandas as pd
import pytest
import torch

from models.registry import REGISTRY, get_spec, load_model
from training.train_lstm import train_lstm
from training.train_neural_ode import train_neural_ode
from training.train_pinn import train_pinn
from utils.config import load_config
from utils.data_loader import AirQualityDataset
from utils.normalization import FEATURES, TARGETS, compute_stats, load_stats

CFG = load_config()
SEQ = 3


@pytest.fixture(scope="module")
def test_split():
    df = pd.read_csv("demo_assets/cleaned_data.csv")
    st = load_stats(CFG["weight_paths"]["normalization"])
    n_train = int(0.9 * len(df))
    x = ((df[FEATURES] - pd.Series(st["x_mean"])) / pd.Series(st["x_std"])).to_numpy("float32")[n_train:]
    y = ((df[TARGETS] - pd.Series(st["y_mean"])) / pd.Series(st["y_std"])).to_numpy("float32")[n_train:]
    windows = torch.tensor(np.stack([x[i:i + SEQ] for i in range(len(x) - SEQ + 1)]))
    return windows, y[SEQ - 1:]


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def predict_with(key, windows, path=None):
    spec = get_spec(key)
    model = load_model(spec, CFG, path or CFG["weight_paths"][spec.weights_key], "cpu")
    with torch.no_grad():
        return spec.forward(model, windows).numpy()


def test_normalization_file_matches_data():
    saved, fresh = load_stats(CFG["weight_paths"]["normalization"]), compute_stats("demo_assets/cleaned_data.csv")
    for k in ("x_mean", "x_std", "y_mean", "y_std"):
        assert saved[k] == pytest.approx(fresh[k])


@pytest.mark.parametrize("key", list(REGISTRY))
def test_nowcast_models_reproduce_saved_metrics(test_split, key):
    x, y = test_split
    saved = pd.read_csv(f"evaluation/{key}/{key}_metrics.csv")["R2"].to_numpy()
    assert r2(y, predict_with(key, x)) == pytest.approx(saved, abs=0.01)


def test_registry_models_all_run_on_a_window():
    x = torch.zeros(4, SEQ, 6)
    for key in REGISTRY:
        spec = get_spec(key)
        assert spec.forward(spec.build(CFG["model"][key]), x).shape == (4, 3)


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError):
        get_spec("transformer")


@pytest.fixture(scope="module")
def tiny_loader(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    df = pd.read_csv("demo_assets/cleaned_data.csv").iloc[:200]
    df[FEATURES].to_csv(d / "X.csv", index=False)
    df[TARGETS].to_csv(d / "y.csv", index=False)
    return torch.utils.data.DataLoader(AirQualityDataset(d / "X.csv", d / "y.csv"), batch_size=32)


def test_training_entry_points_run(tiny_loader):
    dev = torch.device("cpu")
    train_lstm(tiny_loader, tiny_loader, 6, 3, dev, hidden_size=8, num_layers=1, dropout=0.0, epochs=1)
    train_neural_ode(tiny_loader, tiny_loader, 6, 3, dev, hidden_size=8, intermediate_size=8, epochs=1)
    train_pinn(tiny_loader, tiny_loader, 6, 3, 1, dev, hidden_size=8, epochs=1)


def test_forecast_horizon_alignment(tmp_path):
    """The target of window i at horizon h is the row h hours after the nowcast target."""
    df = pd.read_csv("demo_assets/cleaned_data.csv").iloc[:50]
    df[FEATURES].to_csv(tmp_path / "X.csv", index=False)
    df[TARGETS].to_csv(tmp_path / "y.csv", index=False)
    now = AirQualityDataset(tmp_path / "X.csv", tmp_path / "y.csv")
    ahead = AirQualityDataset(tmp_path / "X.csv", tmp_path / "y.csv", horizon=3)
    assert len(ahead) == len(now) - 3
    assert torch.equal(ahead[0][0], now[0][0]) and torch.equal(ahead[0][1], now[3][1])
    with pytest.raises(ValueError):
        AirQualityDataset(tmp_path / "X.csv", tmp_path / "y.csv", horizon=-1)


@pytest.mark.parametrize("h", [24, 48, 72])
@pytest.mark.parametrize("key", list(REGISTRY))
def test_forecast_models_reproduce_saved_metrics(test_split, h, key):
    import os
    sfx = f"_h{h}"
    path = CFG["weight_paths"][get_spec(key).weights_key].replace(".pth", sfx + ".pth")
    metrics_csv = f"evaluation/{key}/{key}_metrics{sfx}.csv"
    if not (os.path.exists(path) and os.path.exists(metrics_csv)):
        pytest.skip("forecast model not trained")
    x, y = test_split
    p = predict_with(key, x[: len(x) - h], path)
    assert r2(y[h:], p) == pytest.approx(pd.read_csv(metrics_csv)["R2"].to_numpy(), abs=0.01)
