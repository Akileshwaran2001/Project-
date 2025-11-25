import torch
from scripts.dataloader import TimeSeriesDataset
from models.nbeats import NBeats
from torch.utils.data import DataLoader
import numpy as np

dataset = TimeSeriesDataset('data/synthetic_multivariate.csv', backcast_length=48, forecast_length=24)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NBeats(input_dim=48, backcast_length=48, forecast_length=24, nb_blocks_per_stack=3, hidden_dim=128, nb_layers=2)
model.to(device)
try:
    model.load_state_dict(torch.load('checkpoints/nbeats.pth', map_location=device))
    print('Loaded checkpoint.')
except FileNotFoundError:
    print('No checkpoint found. Run training first.')
model.eval()
preds = []
trues = []
with torch.no_grad():
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)
mse = np.mean((preds - trues)**2)
print(f'MSE on dataset: {mse:.6f}')
