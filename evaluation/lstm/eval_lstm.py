import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_loader import get_data_loaders  # adjust import if needed
from models.lstm import LSTMRegressor    # adjust import if needed

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def evaluate_model(model_path, config, device):
    # --- Load test data ---
    _, test_loader = get_data_loaders(config, batch_size=config['training']['batch_size'])

    # --- Load model configuration ---
    lstm_cfg = config['model']['lstm']
    input_size = lstm_cfg['input_size']
    output_size = lstm_cfg['output_size']
    hidden_size = lstm_cfg['hidden_size']
    num_layers = lstm_cfg['num_layers']
    dropout = lstm_cfg['dropout']

    # --- Initialize model ---
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    ).to(device)

    # --- Load trained weights ---
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, actuals = [], []

    # --- Inference loop ---
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    preds = np.array(preds)  # shape: (num_samples, 3)
    actuals = np.array(actuals)  # shape: (num_samples, 3)


    mae = [mean_absolute_error(actuals[:, i], preds[:, i]) for i in range(output_size)]
    rmse = [np.sqrt(mean_squared_error(actuals[:, i], preds[:, i])) for i in range(output_size)]
    r2 = [r2_score(actuals[:, i], preds[:, i]) for i in range(output_size)]


    # --- Ensure directories exist before saving ---
    metrics_dir = config['evaluation']['metrics_path']['lstm_metrics']
    preds_dir = config['evaluation']['predictions_path']['lstm_predictions']

    # --- Save metrics ---
    metrics_path = os.path.join(metrics_dir, "lstm_metrics.csv")
    results_path = os.path.join(preds_dir, "lstm_predictions.csv")

    metrics_df = pd.DataFrame({
    'Parameter': ['CO', 'NOx', 'NO2'],
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2})
    metrics_df.to_csv(metrics_path, index=False)

    param_names = ['CO', 'NOx', 'NO2']
    df_actuals = pd.DataFrame(actuals, columns=param_names)
    df_preds = pd.DataFrame(preds, columns=param_names)
    df_results = pd.concat([df_actuals.add_suffix('_actual'), df_preds.add_suffix('_predicted')], axis=1)
    df_results.to_csv(results_path, index=False)
    
    print(f"Metrics saved to {metrics_path}: \n{metrics_df}")
    print(f"\nPredictions saved to {results_path}: \n{df_results.head()}")


'''
def forecast_future(config, model_path, device, forecast_steps=6):
    """
    Forecast multiple future steps (e.g., hours) using recursive prediction.
    """
    # --- Load model config ---
    lstm_cfg = config['model']['lstm']
    model = LSTMRegressor(
        input_size=lstm_cfg['input_size'],
        hidden_size=lstm_cfg['hidden_size'],
        num_layers=lstm_cfg['num_layers'],
        output_size=lstm_cfg['output_size'],
        dropout=lstm_cfg['dropout']
    ).to(device)

    # --- Load model weights ---
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Get test data for last known sequence ---
    _, test_loader = get_data_loaders(config, batch_size=1)
    last_batch = None
    for X_batch, _ in test_loader:
        last_batch = X_batch  # keep last sequence
    input_seq = last_batch.to(device)  # shape: (1, seq_len, input_size)

    preds = []

    with torch.no_grad():
        for _ in range(forecast_steps):
            output = model(input_seq)  # (1, output_size)
            preds.append(output.cpu().numpy())

            # update sequence: drop oldest timestep, append new prediction
            new_input = torch.cat(
                [input_seq[:, 1:, :], output.unsqueeze(1)], dim=1
            )
            input_seq = new_input

    preds = np.array(preds).reshape(forecast_steps, lstm_cfg['output_size'])
    df_forecast = pd.DataFrame(preds, columns=['CO', 'NOx', 'NO2'])
    print(df_forecast)

    # --- Optional: Save forecast results ---
    df_forecast.to_csv(os.path.join(results_path, f'lstm_future_forecast.csv'), index=False)
    print(f"Forecasted {forecast_steps} steps saved to {results_path}")

    return df_forecast'''
