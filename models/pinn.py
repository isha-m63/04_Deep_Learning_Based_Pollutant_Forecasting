import torch
import torch.nn as nn
from torch.autograd import grad

#Model the simplified temporal evolution of pollutant concentrations C(t) of NOx, NO2 and CO  influenced by 
#meteorological variables (temperature, relative humidity and absolute humidity) through an ODE-like relation:
#dC/dt = R(C, meteo) where R is a reaction rate function approximated by the fully connected ANN, embedding physical constraints in the loss.


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

    def physics_loss(self, inputs, outputs, t):
        # inputs include concentrations + meteorological variables
        # outputs = model(inputs) = predicted concentrations at time t

        # Compute d(outputs)/dt via autograd
        outputs_t = grad(outputs, t, 
                         grad_outputs=torch.ones_like(outputs),
                         create_graph=True, retain_graph=True)[0]

        # Define physics residual as derivative minus model output (reaction rates)
        # PINN enforces dC/dt = R(C, meteo) modeled by NN output
        residual = outputs_t - outputs
        
        return torch.mean(residual**2)
