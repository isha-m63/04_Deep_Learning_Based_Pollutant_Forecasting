"""Weights load, saved metrics reproduce, and every training entry point runs for one epoch."""
import numpy as np
import pandas as pd
import pytest
import torch

from models.lstm import LSTMRegressor
from models.neural_ode import NeuralODERegressor
from models.pinn import PINNRegressor
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


def load(cls, key, *args):
    m = cls(*args)
    m.load_state_dict(torch.load(CFG["weight_paths"][key], map_location="cpu", weights_only=True))
    return m.eval()


def test_normalization_file_matches_data():
    saved, fresh = load_stats(CFG["weight_paths"]["normalization"]), compute_stats("demo_assets/cleaned_data.csv")
    for k in ("x_mean", "x_std", "y_mean", "y_std"):
        assert saved[k] == pytest.approx(fresh[k])


def test_lstm_reproduces_saved_metrics(test_split):
    x, y = test_split
    with torch.no_grad():
        p = load(LSTMRegressor, "lstm_model", 6, 64, 2, 3, 0.2)(x).numpy()
    saved = pd.read_csv("evaluation/lstm/lstm_metrics.csv")["R2"].to_numpy()
    assert r2(y, p) == pytest.approx(saved, abs=0.01)


def test_neural_ode_reproduces_saved_metrics(test_split):
    x, y = test_split
    with torch.no_grad():
        p = load(NeuralODERegressor, "neural_ode_model", 6, 64, 64, 3)(x[:, -1], torch.linspace(0, 1, 10)).numpy()
    saved = pd.read_csv("evaluation/neural_ode/neural_ode_metrics.csv")["R2"].to_numpy()
    assert r2(y, p) == pytest.approx(saved, abs=0.01)


def test_pinn_reproduces_saved_metrics(test_split):
    x, y = test_split
    with torch.no_grad():
        p = load(PINNRegressor, "pinn_model", 6, 64, 3, 2)(x[:, -1]).numpy()
    saved = pd.read_csv("evaluation/pinn/pinn_metrics.csv")["R2"].to_numpy()
    assert r2(y, p) == pytest.approx(saved, abs=0.01)


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
