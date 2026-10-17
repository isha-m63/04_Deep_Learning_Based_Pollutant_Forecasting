import torch
import torch.nn as nn
import torch.optim as optim
import timeit
import os
import numpy as np
import pandas as pd
from models.lstm import LSTMRegressor
from utils.data_loader import get_data_loaders

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"



def train_lstm(train_loader, test_loader, input_size, output_size,
               device, hidden_size, num_layers, dropout,
               epochs=50, lr=1e-3, save_path=None, losses_save_path=None):
    
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    start_time = timeit.default_timer()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        avg_val_loss = val_loss / len(test_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    total_time = timeit.default_timer() - start_time
    total_time_in_mins = total_time / 60

    print(f"Total training time: {total_time:.2f} seconds / {total_time_in_mins:.2f} minutes")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    if losses_save_path:
        os.makedirs(losses_save_path, exist_ok=True)
        df_losses = pd.DataFrame({
            'epoch': range(1, epochs + 1),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        df_losses.to_csv(os.path.join(losses_save_path, 'lstm_loss_epoch.csv'), index=False)
        print(f"Train and validation losses saved to {losses_save_path}lstm_loss_epoch.csv")


if __name__ == '__main__':
    from utils.config import load_config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config()
    m = config['model']['lstm']
    t = config['training']

    train_loader, test_loader = get_data_loaders(config, batch_size=t['batch_size'])

    train_lstm(train_loader, test_loader, m['input_size'], m['output_size'], device,
               hidden_size=m['hidden_size'], num_layers=m['num_layers'], dropout=m['dropout'],
               epochs=t['epochs'], lr=t['learning_rate'],
               save_path=config['weight_paths']['lstm_model'],
               losses_save_path=config['evaluation']['loss_epoch_path']['lstm_losses'])
