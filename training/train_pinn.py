import torch
import torch.nn as nn
import torch.optim as optim
import timeit
import os
import pandas as pd
from models.pinn import PINNRegressor  # assume pinn class is in model_pinn.py

def train_pinn(train_loader, test_loader, input_size, output_size, num_layers,
               device, hidden_size, epochs=50, lr=1e-3,
               save_path=None, losses_save_path=None,
               lambda_phys=1.0, delta_t=1.0):  # delta_t is time step (e.g. 1 hour)

    model = PINNRegressor(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    start_time = timeit.default_timer()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Take only last step of inputs for target prediction
            if inputs.ndim == 3:
                inputs_last = inputs[:, -1, :]  # (batch_size, feat_dim)
                inputs_prev = inputs[:, -2, :]  # previous time step for difference approx
            else:
                inputs_last = inputs

            optimizer.zero_grad()

            outputs_last = model(inputs_last)  # output at t
            outputs_prev = model(inputs_prev) if inputs.ndim == 3 else outputs_last  # output at t-dt

            if targets.ndim > 2:
                targets = targets.squeeze()

            # Data loss between predicted output and target
            data_loss = criterion(outputs_last, targets)

            # Physics loss as finite difference approx of dC/dt
            dC_dt_approx = (outputs_last - outputs_prev) / delta_t

            # Physics residual: difference between approx derivative and predicted rate (could be outputs or another network output)
            # Here, for example, you enforce smoothness or zero acceleration as proxy physics constraint
            physics_residual = dC_dt_approx  # if reaction term known, replace with (dC/dt - R(C, params))

            physics_loss = torch.mean(physics_residual**2)

            loss = data_loss + lambda_phys * physics_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs_last.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                if inputs.ndim == 3:
                    inputs_last = inputs[:, -1, :]
                else:
                    inputs_last = inputs

                outputs = model(inputs_last)

                if targets.ndim > 2:
                    targets = targets.squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs_last.size(0)

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
        df_losses.to_csv(os.path.join(losses_save_path, 'pinn_loss_epoch.csv'), index=False)
        print(f"Train/validation losses saved to {os.path.join(losses_save_path, 'pinn_loss_epoch.csv')}")