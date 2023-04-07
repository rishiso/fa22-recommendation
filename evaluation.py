import numpy as np
import pandas as pd

def root_mean_squared_error(truth: pd.core.series.Series, prediction: np.ndarray):
    assert truth.size == prediction.shape[0], "Truth and Prediction must have same number of entries" 
    truth = truth.to_numpy()
    prediction = [_[0] for _ in prediction]
    return np.sqrt(np.mean((truth - prediction)**2))

def mean_absolute_error(truth: pd.core.series.Series, prediction: np.ndarray):
    assert truth.size == prediction.shape[0], "Truth and Prediction must have same number of entries" 
    truth = truth.to_numpy()
    prediction = [_[0] for _ in prediction]
    return np.mean(abs(truth - prediction)); 

