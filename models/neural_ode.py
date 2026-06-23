import torch
import torch.nn as nn
#!pip install torchdiffeq
from torchdiffeq import odeint

import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, t, x):
        return self.net(x)

class NeuralODERegressor(nn.Module):
    def __init__(self, input_size, hidden_size, intermediate_size, output_size):
        super(NeuralODERegressor, self).__init__()
        self.hidden_size = hidden_size

        # initial projection layer (like embedding input features to hidden dim)
        self.fc_in = nn.Linear(input_size, hidden_size)

        # neural ODE function defining dh/dt
        self.odefunc = ODEFunc(hidden_size, intermediate_size)

        # output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, t):
        # x: (batch, input_size)
        z0 = self.fc_in(x)  # initial hidden state
        z_t = odeint(self.odefunc, z0, t)  # integrate over time points t
        # z_t shape: (time_steps, batch, hidden_size)

        out = self.fc_out(z_t[-1])  # take the last time step's hidden state for output
        return out

