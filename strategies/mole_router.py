from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from .rl_qms import BaseStrategy, StrategyOutput


class MoLERouterStrategy(BaseStrategy):
    """
    MoLE Router-only baseline（改进版）:
    - Experts are frozen model predictions P[:, i]
    - Router (2-layer MLP) outputs softmax weights from timestamp-only features
    
    改进点：
    1. active_models: 候选池约束，只在活跃模型中加权
    2. temperature: 温度参数，控制 softmax 的锐度（>1 更平滑，<1 更尖锐）
    3. temporal_holdout: 时间留出验证比例，用于早停和防止过拟合
    4. weight_clip: 权重裁剪，防止单一模型权重过大
    """

    name = "mole_router"

    def __init__(
        self,
        hidden_dim: int = 16,
        epochs: int = 200,
        lr: float = 1e-2,
        batch_size: int = 256,
        head_dropout: float = 0.1,
        eps: float = 1e-8,
        seed: int = 0,
        # 新增参数
        active_models: list = None,    # 候选池约束
        temperature: float = 1.0,      # 温度参数（>1 更平滑）
        temporal_holdout: float = 0.2, # 时间留出验证比例
        early_stop_patience: int = 20, # 早停耐心
        weight_clip: float = 0.8,      # 单模型权重上限
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.head_dropout = head_dropout
        self.eps = eps
        self.seed = seed
        
        # 新增参数
        self.active_models = active_models
        self.temperature = temperature
        self.temporal_holdout = temporal_holdout
        self.early_stop_patience = early_stop_patience
        self.weight_clip = weight_clip

        self._col_mean = None
        self._feature_names: List[str] = []
        self._embedding_spec: Dict[str, Any] = {}
        self._use_uniform = False
        self._params = None
        self._train_rmse = None
        self._val_rmse = None
        self._active_indices = None
        self._active_to_full = None

    def fit(self, P_val, y_val, ctx_val=None, model_names=None):
        """在验证集上训练 Router
        
        Args:
            P_val: 验证集预测矩阵 [N, M]
            y_val: 验证集真值
            ctx_val: 上下文特征（DataFrame 或 None）
            model_names: 模型名列表，用于匹配 active_models
        """
        if P_val is None or y_val is None:
            return self
        P_val = self._fill_nan(P_val, is_fit=True)
        y_val = np.asarray(y_val, dtype=float)
        if P_val.ndim == 1:
            P_val = P_val[:, None]
        if len(P_val) == 0 or len(P_val) != len(y_val):
            return self
        
        M = P_val.shape[1]
        
        # 处理候选池约束
        if self.active_models is not None and model_names is not None:
            self._active_indices = [i for i, m in enumerate(model_names) if m in self.active_models]
            if not self._active_indices:
                self._active_indices = list(range(M))
            self._active_to_full = {i: idx for i, idx in enumerate(self._active_indices)}
        else:
            self._active_indices = list(range(M))
            self._active_to_full = {i: i for i in range(M)}
        
        # 只用活跃模型训练
        P_active = P_val[:, self._active_indices]

        X_val = self._build_time_features(ctx_val, use_existing_spec=False)
        if X_val is None or X_val.shape[1] == 0:
            self._use_uniform = True
            return self

        # 时间留出验证：用后 temporal_holdout 比例的数据做验证
        n_total = len(P_active)
        n_holdout = int(n_total * self.temporal_holdout)
        if n_holdout > 0 and n_total - n_holdout >= 50:
            X_train, X_hold = X_val[:-n_holdout], X_val[-n_holdout:]
            P_train, P_hold = P_active[:-n_holdout], P_active[-n_holdout:]
            y_train, y_hold = y_val[:-n_holdout], y_val[-n_holdout:]
        else:
            X_train, X_hold = X_val, None
            P_train, P_hold = P_active, None
            y_train, y_hold = y_val, None

        self._train_router(X_train, P_train, y_train, X_hold, P_hold, y_hold)
        
        # 计算训练集 RMSE
        y_hat, _ = self._predict_with_router(X_val, P_active)
        self._train_rmse = float(np.sqrt(np.nanmean((y_hat - y_val) ** 2)))
        return self

    def predict(self, P_test, y_test=None, ctx_test=None, model_names=None) -> StrategyOutput:
        P_test_orig = self._fill_nan(P_test, is_fit=False)
        if P_test_orig.ndim == 1:
            P_test_orig = P_test_orig[:, None]
        
        M_full = P_test_orig.shape[1]
        
        # 处理候选池约束
        if self._active_indices is None:
            self._active_indices = list(range(M_full))
            self._active_to_full = {i: i for i in range(M_full)}
        
        P_test = P_test_orig[:, self._active_indices]
        M_active = len(self._active_indices)
        
        use_existing = bool(self._feature_names)
        X_test = self._build_time_features(ctx_test, use_existing_spec=use_existing)
        
        if self._use_uniform or X_test is None or X_test.shape[1] == 0 or self._params is None:
            # 回退到等权平均
            weights_active = np.ones((len(P_test), M_active), dtype=np.float32) / M_active if M_active > 0 else np.zeros((len(P_test), 0))
            y_pred = np.sum(P_test * weights_active, axis=1).astype(np.float32) if M_active > 0 else np.zeros((len(P_test),), dtype=np.float32)
        else:
            y_pred, weights_active = self._predict_with_router(X_test, P_test)
        
        # 将活跃模型的权重映射回全模型空间
        weights_full = np.zeros((len(P_test), M_full), dtype=np.float32)
        for i_active, i_full in self._active_to_full.items():
            weights_full[:, i_full] = weights_active[:, i_active]

        meta = {
            "feature_mode": "timestamp_only",
            "embedding_spec": self._embedding_spec,
            "feature_names": self._feature_names,
            "router": {
                "hidden_dim": self.hidden_dim,
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "activation": "tanh",
                "head_dropout": self.head_dropout,
                "temperature": self.temperature,
                "weight_clip": self.weight_clip,
                "seed": self.seed,
            },
            "active_models_count": M_active,
            "total_models_count": M_full,
            "train_rmse": self._train_rmse,
            "val_rmse": self._val_rmse,
            "cost_assumption": "posthoc_weighting_or_switching",
        }
        return StrategyOutput(y_pred, weights_full, meta)

    def _fill_nan(self, P, is_fit: bool) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        if P.ndim == 1:
            P = P[:, None]
        if is_fit:
            col_mean = np.nanmean(P, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            self._col_mean = col_mean
        if self._col_mean is not None:
            P = np.where(np.isfinite(P), P, self._col_mean)
        return P

    def _build_time_features(self, ctx, use_existing_spec: bool = False) -> Optional[np.ndarray]:
        if ctx is None:
            self._feature_names = []
            self._embedding_spec = {}
            return None

        df = ctx if isinstance(ctx, pd.DataFrame) else pd.DataFrame(ctx)
        ts = None
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")

        feature_map: Dict[str, np.ndarray] = {}
        spec: Dict[str, Any] = {}

        def _add_feature(name: str, values: np.ndarray, norm_desc: str):
            feature_map[name] = values.astype(np.float32)
            spec[name] = {"norm": norm_desc}

        if ts is not None and ts.notna().any():
            hour = ts.dt.hour.to_numpy()
            _add_feature("hour", hour / 23.0 - 0.5, "hour/23-0.5")

            dow = ts.dt.dayofweek.to_numpy()
            _add_feature("dayofweek", dow / 6.0 - 0.5, "dow/6-0.5")

            month = ts.dt.month.to_numpy()
            _add_feature("month", (month - 1) / 11.0 - 0.5, "(month-1)/11-0.5")

            doy = ts.dt.dayofyear.to_numpy()
            _add_feature("dayofyear", (doy - 1) / 365.0 - 0.5, "(doy-1)/365-0.5")

            is_weekend = (dow >= 5).astype(float)
            _add_feature("is_weekend", is_weekend - 0.5, "weekend-0.5")
        else:
            for col, norm_desc, scale, shift in [
                ("ctx_hour", "hour/23-0.5", 23.0, 0.5),
                ("ctx_dayofweek", "dow/6-0.5", 6.0, 0.5),
                ("ctx_month", "(month-1)/11-0.5", 11.0, 0.5),
                ("ctx_dayofyear", "(doy-1)/365-0.5", 365.0, 0.5),
            ]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
                    if col == "ctx_month":
                        vals = (vals - 1) / scale - shift
                    else:
                        vals = vals / scale - shift
                    _add_feature(col.replace("ctx_", ""), vals, norm_desc)

        if "ctx_is_holiday" in df.columns:
            hol = pd.to_numeric(df["ctx_is_holiday"], errors="coerce").fillna(0).to_numpy()
            _add_feature("is_holiday", hol - 0.5, "holiday-0.5")

        if "ctx_is_weekend" in df.columns and "is_weekend" not in feature_map:
            weekend = pd.to_numeric(df["ctx_is_weekend"], errors="coerce").fillna(0).to_numpy()
            _add_feature("is_weekend", weekend - 0.5, "weekend-0.5")

        if not feature_map:
            self._feature_names = []
            self._embedding_spec = {}
            return None

        if use_existing_spec and self._feature_names:
            feats = []
            for name in self._feature_names:
                if name in feature_map:
                    feats.append(feature_map[name])
                else:
                    feats.append(np.zeros(len(df), dtype=np.float32))
            X = np.vstack(feats).T
            return np.nan_to_num(X, nan=0.0)

        names = list(feature_map.keys())
        X = np.vstack([feature_map[n] for n in names]).T
        X = np.nan_to_num(X, nan=0.0)
        self._feature_names = names
        self._embedding_spec = spec
        return X

    def _init_params(self, in_dim: int, out_dim: int):
        rng = np.random.default_rng(self.seed)
        W1 = rng.normal(0, 0.1, size=(in_dim, self.hidden_dim)).astype(np.float32)
        b1 = np.zeros((self.hidden_dim,), dtype=np.float32)
        W2 = rng.normal(0, 0.1, size=(self.hidden_dim, out_dim)).astype(np.float32)
        b2 = np.zeros((out_dim,), dtype=np.float32)
        self._params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def _softmax(self, z: np.ndarray, temperature: float = None) -> np.ndarray:
        """带温度参数的 softmax
        
        Args:
            z: logits [N, M]
            temperature: 温度参数（>1 更平滑，<1 更尖锐）
        
        Returns:
            权重矩阵 [N, M]
        """
        if temperature is None:
            temperature = self.temperature
        z = z / max(temperature, 1e-6)  # 温度缩放
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        w = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + self.eps)
        
        # 权重裁剪：防止单一模型权重过大
        if self.weight_clip < 1.0:
            w = np.clip(w, 0, self.weight_clip)
            w = w / (np.sum(w, axis=1, keepdims=True) + self.eps)
        
        return w

    def _train_router(self, X: np.ndarray, P: np.ndarray, y: np.ndarray,
                      X_val: np.ndarray = None, P_val: np.ndarray = None, y_val: np.ndarray = None):
        """训练 Router 网络
        
        Args:
            X: 训练特征 [N_train, D]
            P: 训练预测矩阵 [N_train, M]
            y: 训练真值 [N_train]
            X_val: 验证特征（用于早停）
            P_val: 验证预测矩阵
            y_val: 验证真值
        """
        T, M = P.shape
        if self._params is None:
            self._init_params(X.shape[1], M)
        params = self._params
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
        rng = np.random.default_rng(self.seed)
        batch_size = max(1, int(self.batch_size))
        
        # 早停相关
        best_val_rmse = float('inf')
        best_params = None
        patience_counter = 0
        use_early_stop = X_val is not None and P_val is not None and y_val is not None

        for epoch in range(self.epochs):
            order = rng.permutation(T)
            for i in range(0, T, batch_size):
                idx = order[i : i + batch_size]
                Xb = X[idx]
                Pb = P[idx]
                yb = y[idx]

                z1 = Xb @ W1 + b1
                h = np.tanh(z1)
                z2 = h @ W2 + b2
                w = self._softmax(z2)

                if self.head_dropout > 0:
                    mask = rng.random(w.shape) >= self.head_dropout
                    all_drop = mask.sum(axis=1) == 0
                    if np.any(all_drop):
                        keep_idx = np.argmax(w[all_drop], axis=1)
                        mask[np.where(all_drop)[0], keep_idx] = True
                    w = w * mask
                    w = w / (np.sum(w, axis=1, keepdims=True) + self.eps)

                y_hat = np.sum(w * Pb, axis=1)
                grad_y = (2.0 * (y_hat - yb)) / max(1, len(yb))
                dZ = grad_y[:, None] * w * (Pb - y_hat[:, None])

                dW2 = h.T @ dZ
                db2 = np.sum(dZ, axis=0)
                dH = dZ @ W2.T
                dA1 = dH * (1.0 - np.tanh(z1) ** 2)
                dW1 = Xb.T @ dA1
                db1 = np.sum(dA1, axis=0)

                W1 -= self.lr * dW1
                b1 -= self.lr * db1
                W2 -= self.lr * dW2
                b2 -= self.lr * db2
            
            # 早停检查
            if use_early_stop and epoch > 0 and epoch % 10 == 0:
                # 临时保存当前参数
                params["W1"], params["b1"], params["W2"], params["b2"] = W1, b1, W2, b2
                self._params = params
                
                y_val_hat, _ = self._predict_with_router(X_val, P_val)
                val_rmse = float(np.sqrt(np.nanmean((y_val_hat - y_val) ** 2)))
                
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_params = {"W1": W1.copy(), "b1": b1.copy(), "W2": W2.copy(), "b2": b2.copy()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience // 10:
                        # 恢复最佳参数
                        if best_params is not None:
                            W1, b1, W2, b2 = best_params["W1"], best_params["b1"], best_params["W2"], best_params["b2"]
                        break

        params["W1"], params["b1"], params["W2"], params["b2"] = W1, b1, W2, b2
        self._params = params
        self._val_rmse = best_val_rmse if use_early_stop else None

    def _predict_with_router(self, X: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        params = self._params
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
        h = np.tanh(X @ W1 + b1)
        logits = h @ W2 + b2
        weights = self._softmax(logits).astype(np.float32)
        y_pred = np.sum(weights * P, axis=1).astype(np.float32)
        return y_pred, weights
