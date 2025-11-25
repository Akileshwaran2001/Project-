import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, backcast_length=48, forecast_length=24):
        df = pd.read_csv(csv_path)
        # use first column as target for simplicity, keep others as covariates if needed
        data = df.values.astype('float32')
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        self.series = data[:,0]  # using first feature as target
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.length = len(self.series) - backcast_length - forecast_length + 1
        self.scaler = scaler

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx
        backcast = self.series[start:start + self.backcast_length]
        forecast = self.series[start + self.backcast_length:start + self.backcast_length + self.forecast_length]
        return torch.from_numpy(backcast).float(), torch.from_numpy(forecast).float()
