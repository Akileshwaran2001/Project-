import numpy as np

def mase(y_true, y_pred, naively_diff=None):
    # simple MASE for seasonal=1 if naively_diff not provided
    if naively_diff is None:
        naively_diff = np.mean(np.abs(np.diff(y_true, n=1)))
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / (naively_diff + 1e-8)
