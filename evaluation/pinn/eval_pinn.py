import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.pinn import PINNRegressor
from utils.data_loader import get_data_loaders


def evaluate_model(model_path, config, device):
    _, test_loader = get_data_loaders(config, batch_size=config['training']['batch_size'])

    cfg = config['model']['pinn']
    output_size = cfg['output_size']
    model = PINNRegressor(cfg['input_size'], cfg['hidden_size'], output_size, cfg['num_layers']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            if X_batch.ndim == 3:                 # the PINN uses only the latest time step
                X_batch = X_batch[:, -1, :]
            preds.extend(model(X_batch).cpu().numpy())
            actuals.extend(y_batch.numpy())

    preds, actuals = np.array(preds), np.array(actuals)
    param_names = ['CO', 'NOx', 'NO2'][:output_size]

    metrics_df = pd.DataFrame({
        'Parameter': param_names,
        'MAE': [mean_absolute_error(actuals[:, i], preds[:, i]) for i in range(output_size)],
        'RMSE': [np.sqrt(mean_squared_error(actuals[:, i], preds[:, i])) for i in range(output_size)],
        'R2': [r2_score(actuals[:, i], preds[:, i]) for i in range(output_size)],
    })
    results_df = pd.concat([
        pd.DataFrame(actuals, columns=param_names).add_suffix('_actual'),
        pd.DataFrame(preds, columns=param_names).add_suffix('_predicted'),
    ], axis=1)

    metrics_dir = config['evaluation']['metrics_path']['pinn_metrics']
    preds_dir = config['evaluation']['predictions_path']['pinn_predictions']
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(metrics_dir, "pinn_metrics.csv"), index=False)
    results_df.to_csv(os.path.join(preds_dir, "pinn_predictions.csv"), index=False)

    print(f"Metrics saved to {metrics_dir}:\n{metrics_df}")
    return metrics_df, results_df


if __name__ == "__main__":
    from utils.config import load_config
    config = load_config()
    evaluate_model(config['weight_paths']['pinn_model'], config, torch.device('cpu'))
