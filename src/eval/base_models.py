from typing import Dict, List
import numpy as np
import pandas as pd
from src.models.registry import model_registry


def train_base_models(models: List[str], X_train: pd.DataFrame, y_train: pd.Series,
                      model_overrides: Dict[str, Dict] = None) -> Dict[str, any]:
    """训练基线模型。"""
    model_overrides = model_overrides or {}
    fitted = {}
    for mid in models:
        try:
            print(f"    -> Training {mid}...", flush=True)
            kwargs = model_overrides.get(mid, {})
            model = model_registry.create(mid, **kwargs)
            model.fit(X_train, y_train)
            fitted[mid] = model
        except Exception as exc:
            print(f"  {mid} fit failed: {exc}")
    return fitted


def predict_models(fitted: Dict[str, any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    preds = {}
    for mid, model in fitted.items():
        try:
            preds[mid] = model.predict(X)
        except Exception as exc:
            print(f"  {mid} predict failed: {exc}")
    return preds
