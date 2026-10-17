"""Shared evaluation: predictions on the held-out split, MAE / RMSE / R2 per pollutant (z-score units), CSV output."""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.registry import ModelSpec, get_spec, load_model
from utils.config import horizon_suffix
from utils.data_loader import get_data_loaders

PARAMETERS = ["CO", "NOx", "NO2"]


def evaluate(spec: ModelSpec, model_path, config: dict, device):
    _, test_loader = get_data_loaders(config, batch_size=config["training"]["batch_size"])
    model = load_model(spec, config, model_path, device)

    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds.extend(spec.forward(model, X_batch.to(device)).cpu().numpy())
            actuals.extend(y_batch.numpy())
    preds, actuals = np.array(preds), np.array(actuals)

    n = actuals.shape[1]
    names = PARAMETERS[:n]
    metrics_df = pd.DataFrame({
        "Parameter": names,
        "MAE": [mean_absolute_error(actuals[:, i], preds[:, i]) for i in range(n)],
        "RMSE": [np.sqrt(mean_squared_error(actuals[:, i], preds[:, i])) for i in range(n)],
        "R2": [r2_score(actuals[:, i], preds[:, i]) for i in range(n)],
    })
    results_df = pd.concat([pd.DataFrame(actuals, columns=names).add_suffix("_actual"),
                            pd.DataFrame(preds, columns=names).add_suffix("_predicted")], axis=1)

    sfx = horizon_suffix(config)
    metrics_dir = config["evaluation"]["metrics_path"][f"{spec.key}_metrics"]
    preds_dir = config["evaluation"]["predictions_path"][f"{spec.key}_predictions"]
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(metrics_dir, f"{spec.key}_metrics{sfx}.csv"), index=False)
    results_df.to_csv(os.path.join(preds_dir, f"{spec.key}_predictions{sfx}.csv"), index=False)

    print(f"Metrics saved to {metrics_dir}:\n{metrics_df}")
    return metrics_df, results_df


def main(key: str) -> None:
    """Command-line entry point shared by evaluation/<model>/eval_<model>.py"""
    import argparse
    from utils.config import load_config, with_horizon

    spec = get_spec(key)
    parser = argparse.ArgumentParser(description=f"Evaluate the {spec.label} model on the held-out test split")
    parser.add_argument("--horizon", type=int, default=0, help="hours ahead that the model predicts (0 = nowcast)")
    config = with_horizon(load_config(), parser.parse_args().horizon)
    weights = config["weight_paths"][spec.weights_key].replace(".pth", horizon_suffix(config) + ".pth")
    evaluate(spec, weights, config, torch.device("cpu"))
