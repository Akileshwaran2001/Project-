import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, nb_layers, theta_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(nb_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, theta_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, backcast_dim, forecast_dim, hidden_dim=256, nb_layers=4, theta_dim=None):
        super().__init__()
        if theta_dim is None:
            theta_dim = backcast_dim + forecast_dim
        self.fc_block = FullyConnectedBlock(input_dim, hidden_dim, nb_layers, theta_dim)
        self.backcast_dim = backcast_dim
        self.forecast_dim = forecast_dim

    def forward(self, x):
        theta = self.fc_block(x)
        backcast = theta[..., :self.backcast_dim]
        forecast = theta[..., self.backcast_dim:self.backcast_dim + self.forecast_dim]
        return backcast, forecast

class NBeats(nn.Module):
    def __init__(self, input_dim, backcast_length, forecast_length, stack_types=('generic',), nb_blocks_per_stack=3,
                 hidden_dim=256, nb_layers=4):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stacks = nn.ModuleList()
        for _ in stack_types:
            for _ in range(nb_blocks_per_stack):
                block = NBeatsBlock(input_dim=backcast_length, backcast_dim=backcast_length,
                                    forecast_dim=forecast_length, hidden_dim=hidden_dim, nb_layers=nb_layers)
                self.stacks.append(block)

    def forward(self, x):
        # x shape: (batch, backcast_length)
        residual = x
        forecast = 0
        for block in self.stacks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast
