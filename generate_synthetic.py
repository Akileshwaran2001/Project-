import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)
np.random.seed(0)

# Generate synthetic multivariate time series (3 channels) with seasonality + trend + noise
timesteps = 2000
t = np.arange(timesteps)
series = []
for i in range(3):
    trend = 0.001*(i+1)*t
    seasonal = 0.5 * np.sin(2 * np.pi * t / (50 + 10*i))
    noise = 0.1 * np.random.randn(timesteps)
    s = 2.0 + trend + seasonal + noise
    series.append(s)
df = pd.DataFrame(np.vstack(series).T, columns=['feat1','feat2','feat3'])
df.to_csv('data/synthetic_multivariate.csv', index=False)
print('Saved data/synthetic_multivariate.csv')
