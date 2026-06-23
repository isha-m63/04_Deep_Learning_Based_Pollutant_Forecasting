# training/train_neural_ode.py
import torch
import torch.nn as nn
import torch.optim as optim
import timeit
import os
import pandas as pd
from models.neural_ode import NeuralODERegressor

def train_neural_ode(train_loader, test_loader, input_size, output_size,
                     device, hidden_size, intermediate_size,
                     epochs=50, lr=1e-3, save_path=None, losses_save_path=None):

    model = NeuralODERegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        output_size=output_size
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    t = torch.linspace(0, 1, 10).to(device)

    start_time = timeit.default_timer()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:   
            inputs, targets = inputs.to(device), targets.to(device)

            # Flatten sequence input (batch, seq_len, input_size) → (batch, input_size)
            # We'll take the last time step, since your y_seq is the last step target
            if inputs.ndim == 3:
                inputs = inputs[:, -1, :]  
            optimizer.zero_grad()
            outputs = model(inputs, t)

            
            if targets.ndim > 2:
                targets = targets.squeeze()

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

                if inputs.ndim == 3:
                    inputs = inputs[:, -1, :]  
                outputs = model(inputs, t)
                if targets.ndim > 2:
                    targets = targets.squeeze()

                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(test_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    total_time = timeit.default_timer() - start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

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
        df_losses.to_csv(os.path.join(losses_save_path, 'neural_ode_loss_epoch.csv'), index=False)
        print(f"Train/validation losses saved to {os.path.join(losses_save_path, 'neural_ode_loss_epoch.csv')}")
