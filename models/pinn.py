import torch
import torch.nn as nn

# A tanh MLP mapping (sensor signals + meteorology) at one time step to CO, NOx and NO2.
# "Physics-informed" here means a soft temporal-smoothness constraint added to the data loss during training
# (see training/train_pinn.py); it is NOT a diffusion-advection PDE: the data comes from a single location.


class PINNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PINNRegressor, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, features)
        return self.net(x)
