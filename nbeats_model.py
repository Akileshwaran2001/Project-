
# nbeats_model.py
import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, theta_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, theta_size)
        )
    def forward(self, x):
        return self.fc(x)

class NBeats(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, theta_size=48):
        super().__init__()
        self.block = NBeatsBlock(input_size, hidden_size, theta_size)
    def forward(self, x):
        theta = self.block(x)
        backcast = theta[..., :x.size(-1)]
        forecast = theta[..., x.size(-1):]
        return backcast, forecast
