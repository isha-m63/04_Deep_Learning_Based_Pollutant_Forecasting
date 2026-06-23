import os
import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_loader import get_data_loaders  
from models.neural_ode import NeuralODERegressor
from torchdiffeq import odeint


def evaluate_model(model_path, config, device):
    
    _, test_loader = get_data_loaders(config, batch_size=config['training']['batch_size'])

    
    neural_ode_cfg = config['model']['neural_ode']
    input_size = neural_ode_cfg['input_size']
    output_size = neural_ode_cfg['output_size']
    hidden_size = neural_ode_cfg['hidden_size']
    intermediate_size = neural_ode_cfg['intermediate_size']

    
    model = NeuralODERegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        output_size=output_size
    ).to(device)

    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, actuals = [], []
    t = torch.linspace(0, 1, 10).to(device)  # same integration range as training

   
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # use last timestep as model input
            if X_batch.ndim == 3:
                X_batch = X_batch[:, -1, :]

            outputs = model(X_batch, t)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

    
    mae = [mean_absolute_error(actuals[:, i], preds[:, i]) for i in range(output_size)]
    rmse = [np.sqrt(mean_squared_error(actuals[:, i], preds[:, i])) for i in range(output_size)]
    r2 = [r2_score(actuals[:, i], preds[:, i]) for i in range(output_size)]

  
    metrics_dir = config['evaluation']['metrics_path']['neural_ode_metrics']
    preds_dir = config['evaluation']['predictions_path']['neural_ode_predictions']

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, "neural_ode_metrics.csv")
    results_path = os.path.join(preds_dir, "neural_ode_predictions.csv")


    param_names = ['CO', 'NOx', 'NO2'][:output_size]  # adjust dynamically if needed

    metrics_df = pd.DataFrame({
        'Parameter': param_names,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })
    metrics_df.to_csv(metrics_path, index=False)

    
    param_names = ['CO', 'NOx', 'NO2']
    df_actuals = pd.DataFrame(actuals, columns=param_names)
    df_preds = pd.DataFrame(preds, columns=param_names)
    df_results = pd.concat([df_actuals.add_suffix('_actual'), df_preds.add_suffix('_predicted')], axis=1)
    df_results.to_csv(results_path, index=False)
    
    print(f"Metrics saved to {metrics_path}: \n{metrics_df}")
    print(f"\nPredictions saved to {results_path}: \n{df_results.head()}")


    return metrics_df, df_results