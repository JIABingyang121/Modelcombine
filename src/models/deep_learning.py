from __future__ import annotations
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import warnings
import os

# Optional backends for Deep Learning models
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
except ImportError:
    AutoModel = None
    AutoConfig = None
    AutoModelForCausalLM = None

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import Informer, Autoformer
except ImportError:
    NeuralForecast = None
    Informer = None
    Autoformer = None

class DeepLearningBaseModel(BaseEstimator, RegressorMixin):
    """Base class for Deep Learning based time series models"""
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.scaler_y = StandardScaler()
        self.device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'

    def _check_torch(self):
        if torch is None:
            raise ImportError("PyTorch is required for this model. Install with: pip install torch")

    def _prepare_df(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Prepare DataFrame for NeuralForecast (ds, y, unique_id)"""
        df = pd.DataFrame()
        # Assuming X index is datetime
        if hasattr(X, 'index') and isinstance(X.index, pd.DatetimeIndex):
            df['ds'] = X.index
        else:
            # Fallback if no datetime index
            df['ds'] = pd.date_range(start='2020-01-01', periods=len(X), freq='H')
        
        df['unique_id'] = '1' # Single series
        
        if y is not None:
            df['y'] = y.values
            
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._check_torch()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_torch()
        return np.zeros(len(X))


class InformerModel(DeepLearningBaseModel):
    """
    Wrapper for Informer model using NeuralForecast.
    """
    def __init__(self, h=24, input_size=96, max_steps=100, **params):
        super().__init__(**params)
        self.h = h # prediction horizon
        self.input_size = input_size
        self.max_steps = max_steps
        self.nf = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._check_torch()
        if NeuralForecast is None:
            raise ImportError("neuralforecast is required. Install with: pip install neuralforecast")
            
        # Prepare data
        train_df = self._prepare_df(X, y)
        
        # Initialize model
        models = [
            Informer(
                h=self.h,
                input_size=self.input_size,
                max_steps=self.max_steps,
                scaler_type='standard',
                **self.params
            )
        ]
        
        self.nf = NeuralForecast(models=models, freq='H') # Assuming Hourly
        self.nf.fit(df=train_df)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("Model must be fitted before prediction")
            
        forecasts = self.nf.predict()
        # forecasts has columns [ds, unique_id, Informer, y]
        
        preds = forecasts['Informer'].values
        
        if len(preds) > len(X):
            return preds[:len(X)]
        elif len(preds) < len(X):
            # Pad with last value
            return np.pad(preds, (0, len(X) - len(preds)), 'edge')
        return preds


class AutoformerModel(DeepLearningBaseModel):
    """
    Wrapper for Autoformer model using NeuralForecast.
    """
    def __init__(self, h=24, input_size=96, max_steps=100, **params):
        super().__init__(**params)
        self.h = h
        self.input_size = input_size
        self.max_steps = max_steps
        self.nf = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._check_torch()
        if NeuralForecast is None:
            raise ImportError("neuralforecast is required. Install with: pip install neuralforecast")
            
        train_df = self._prepare_df(X, y)
        
        models = [
            Autoformer(
                h=self.h,
                input_size=self.input_size,
                max_steps=self.max_steps,
                scaler_type='standard',
                **self.params
            )
        ]
        
        self.nf = NeuralForecast(models=models, freq='H')
        self.nf.fit(df=train_df)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("Model must be fitted before prediction")
        
        forecasts = self.nf.predict()
        preds = forecasts['Autoformer'].values
        
        if len(preds) > len(X):
            return preds[:len(X)]
        elif len(preds) < len(X):
            return np.pad(preds, (0, len(X) - len(preds)), 'edge')
        return preds


class PowerGPTModel(DeepLearningBaseModel):
    """
    Wrapper for PowerGPT or similar Large Language Models adapted for Time Series.
    Could be based on GPT-2/Llama architecture fine-tuned on power data.
    """
    def __init__(self, model_name="gpt2", **params):
        super().__init__(**params)
        self.model_name = model_name

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._check_torch()
        if AutoModel is None:
             raise ImportError("transformers library is required. Install with: pip install transformers")
             
        print(f"Initializing PowerGPT based on {self.model_name}")
        # self.model = AutoModel.from_pretrained(self.model_name)
        # Fine-tuning logic would go here
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("Model must be fitted before prediction")
        return np.zeros(len(X))
