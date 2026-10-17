"""
Z-score statistics used by notebooks/preprocessing.ipynb (first 90 % of cleaned_data.csv,
pandas std with ddof=1). Saved next to the weights so inference does not have to recompute them.

    python -m utils.normalization          # writes weights/normalization.json
"""
import json
import os

import pandas as pd

FEATURES = ["sensor_co", "sensor_nox", "sensor_no2", "temperature", "relative_humidity", "absolute_humidity"]
TARGETS = ["co_ground_truth", "nox_ground_truth", "no2_ground_truth"]
TRAIN_FRACTION = 0.9


def compute_stats(cleaned_csv: str) -> dict:
    df = pd.read_csv(cleaned_csv)
    train = df.iloc[: int(TRAIN_FRACTION * len(df))]
    return {
        "features": FEATURES, "targets": TARGETS, "train_fraction": TRAIN_FRACTION,
        "x_mean": train[FEATURES].mean().to_dict(), "x_std": train[FEATURES].std().to_dict(),
        "y_mean": train[TARGETS].mean().to_dict(), "y_std": train[TARGETS].std().to_dict(),
    }


def save_stats(cleaned_csv: str, out_path: str) -> dict:
    stats = compute_stats(cleaned_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    from utils.config import load_config
    cfg = load_config()
    save_stats(os.path.join(cfg["data"]["processed"], "cleaned_data.csv"), cfg["weight_paths"]["normalization"])
    print("saved", cfg["weight_paths"]["normalization"])
