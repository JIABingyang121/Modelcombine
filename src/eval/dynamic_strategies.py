from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from src.eval.metrics import compute_extreme_weights, robust_mae, ROBUST_CFG


class ScenarioGatingNetwork:
    """
    场景门控网络（改进版）：根据场景特征预测每个样本的模型权重
    """
    def __init__(self, model_cols: List[str], method: str = "ridge", top_m: int = None,
                 robust_cfg: Dict[str, float] | None = None,
                 active_models: List[str] = None,
                 fallback_model: str = None,
                 corr_penalty_enabled: bool = False,
                 corr_penalty_scale: float = 0.2):
        self.model_cols = model_cols
        self.method = method
        self.error_predictors = {}
        self.scaler = StandardScaler()
        self.ctx_cols = []
        self.top_m = top_m
        self.robust_cfg = robust_cfg or ROBUST_CFG
        self.active_models = active_models
        self.fallback_model = fallback_model
        self._effective_models = None
        self._fallback_weights = None
        self.corr_penalty_enabled = corr_penalty_enabled
        self.corr_penalty_scale = corr_penalty_scale
        self._corr_penalty_factors: Dict[str, float] = {}

    def _get_context_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("ctx_")]

    def _get_context_features(self, df: pd.DataFrame) -> np.ndarray:
        features_df = pd.DataFrame(index=df.index)
        if self.ctx_cols:
            features_df = df[self.ctx_cols].fillna(0).copy()

        # 漂移代理特征：模型分歧度越大，通常意味着当前场景不稳定/分布偏移更强。
        # 训练和预测阶段都追加相同维度，保证 scaler 维度一致。
        if self._effective_models is not None and len(self._effective_models) > 1:
            # 使用训练期固定模型集合；缺失列按行均值填充，保证语义与维度稳定。
            preds = pd.DataFrame(index=df.index)
            for m in self._effective_models:
                if m in df.columns:
                    preds[m] = pd.to_numeric(df[m], errors="coerce")
                else:
                    preds[m] = np.nan

            row_mean = preds.mean(axis=1)
            preds = preds.T.fillna(row_mean).T.fillna(0.0)
            features_df["ctx_pred_std"] = preds.std(axis=1).fillna(0.0)
            features_df["ctx_pred_range"] = (preds.max(axis=1) - preds.min(axis=1)).fillna(0.0)

        if features_df.shape[1] == 0:
            return np.zeros((len(df), 0))
        return features_df.values.astype(float)

    def _compute_corr_penalty_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        if "y" not in df.columns or not self._effective_models:
            return {}
        y = df["y"].values
        n_models = len(self._effective_models)
        if n_models <= 1:
            return {m: 1.0 for m in self._effective_models}
        errors = []
        for m in self._effective_models:
            if m not in df.columns:
                return {}
            errors.append(df[m].values - y)
        err_mat = np.column_stack(errors)
        corr = np.corrcoef(err_mat.T)
        corr = np.nan_to_num(corr, nan=1.0, posinf=1.0, neginf=1.0)
        mean_abs_corr = []
        for i in range(n_models):
            vals = [abs(float(corr[i, j])) for j in range(n_models) if j != i]
            mean_abs_corr.append(float(np.mean(vals)) if vals else 0.0)
        penalty_factor = np.exp(-float(self.corr_penalty_scale) * np.asarray(mean_abs_corr))
        return {m: float(penalty_factor[i]) for i, m in enumerate(self._effective_models)}

    def fit(self, val_df: pd.DataFrame):
        if self.active_models is not None:
            self._effective_models = [m for m in self.active_models if m in self.model_cols and m in val_df.columns]
            if not self._effective_models:
                self._effective_models = [m for m in self.model_cols if m in val_df.columns]
        else:
            self._effective_models = [m for m in self.model_cols if m in val_df.columns]
        if self.corr_penalty_enabled:
            self._corr_penalty_factors = self._compute_corr_penalty_factors(val_df)
        else:
            self._corr_penalty_factors = {m: 1.0 for m in self._effective_models}

        self.ctx_cols = self._get_context_cols(val_df)
        context_features = self._get_context_features(val_df)

        y_true = val_df["y"].values

        model_maes = {}
        for m in self._effective_models:
            model_maes[m] = float(mean_absolute_error(y_true, val_df[m].values))
        if model_maes:
            sorted_models = sorted(model_maes.items(), key=lambda x: x[1])
            best_k = sorted_models[:min(self.top_m or 3, len(sorted_models))]
            inv_maes = {m: 1.0 / (mae + 1e-6) for m, mae in best_k}
            inv_sum = sum(inv_maes.values())
            self._fallback_weights = {m: inv_maes[m] / inv_sum for m in inv_maes}
        if self.corr_penalty_enabled:
            self._corr_penalty_factors = self._compute_corr_penalty_factors(val_df)
        else:
            self._corr_penalty_factors = {m: 1.0 for m in self._effective_models}

        if context_features.shape[1] == 0:
            print("    [GatingNetwork] 无场景特征，使用回退权重")
            return False

        self.scaler.fit(context_features)
        X_scaled = self.scaler.transform(context_features)

        from src.utils.blocked_cv import blocked_cv_select_alpha as _bcv_alpha

        for m in self._effective_models:
            if m not in val_df.columns:
                continue
            abs_errors = np.abs(val_df[m].values - y_true)
            if self.robust_cfg.get("enable", False):
                weights = compute_extreme_weights(y_true, abs_errors, self.robust_cfg)
                abs_errors = abs_errors * weights
            clip_thr = np.percentile(abs_errors, 99) if len(abs_errors) > 0 else 0
            abs_errors = np.clip(abs_errors, 0, clip_thr)
            target = np.log1p(abs_errors)

            if self.method == "mlp":
                predictor = MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    max_iter=500,
                    early_stopping=True,
                    random_state=42
                )
            else:
                _best_ridge_alpha, _ = _bcv_alpha(
                    X_scaled, target,
                    alphas=[1.0, 10.0, 100.0],
                    n_folds=3, min_train=30,
                    positive=False, fit_intercept=True,
                )
                predictor = Ridge(alpha=_best_ridge_alpha)

            try:
                predictor.fit(X_scaled, target)
                self.error_predictors[m] = (predictor, clip_thr)
            except Exception as e:
                print(f"    [GatingNetwork] {m} 误差预测器训练失败: {e}")

        print(f"    [GatingNetwork] 训练了 {len(self.error_predictors)} 个误差预测器 (method={self.method})")
        return len(self.error_predictors) > 0

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        n_samples = len(test_df)
        effective_models = self._effective_models or self.model_cols

        if not self.error_predictors:
            if self._fallback_weights:
                predictions = np.zeros(n_samples)
                for m, w in self._fallback_weights.items():
                    if m in test_df.columns:
                        predictions += test_df[m].values * w
                return predictions, float(len(self._fallback_weights))
            return test_df[effective_models].mean(axis=1).values, float(len(effective_models))

        context_features = self._get_context_features(test_df)
        if context_features.shape[1] == 0:
            if self._fallback_weights:
                predictions = np.zeros(n_samples)
                for m, w in self._fallback_weights.items():
                    if m in test_df.columns:
                        predictions += test_df[m].values * w
                return predictions, float(len(self._fallback_weights))
            return test_df[effective_models].mean(axis=1).values, float(len(effective_models))

        X_scaled = self.scaler.transform(context_features)

        predicted_errors = {}
        for m, pack in self.error_predictors.items():
            predictor, clip_thr = pack
            pred_log = predictor.predict(X_scaled)
            pred_err = np.expm1(pred_log)
            pred_err = np.clip(pred_err, 1e-6, clip_thr)
            predicted_errors[m] = pred_err

        predictions = np.zeros(n_samples)
        models_used = []
        for i in range(n_samples):
            inv_sum = 0
            weighted_sum = 0
            order = sorted(predicted_errors.items(), key=lambda kv: kv[1][i])
            selected = order if self.top_m is None else order[: self.top_m]
            for m, pred_err_arr in selected:
                inv_err = 1.0 / pred_err_arr[i]
                if self.corr_penalty_enabled:
                    inv_err *= float(self._corr_penalty_factors.get(m, 1.0))
                weighted_sum += test_df[m].iloc[i] * inv_err
                inv_sum += inv_err
            if inv_sum > 0:
                predictions[i] = weighted_sum / inv_sum
            else:
                predictions[i] = test_df[self.model_cols].iloc[i].mean()
            models_used.append(len(selected))
        avg_used = float(np.mean(models_used)) if models_used else float(len(self.model_cols))
        return predictions, avg_used


class ScenarioBucketSelector:
    """
    场景分桶选择器（改进版）：把样本分到不同场景桶，每个桶内学习模型权重
    """
    def __init__(self, model_cols: List[str], min_bucket_size: int = 30,
                 robust_cfg: Dict[str, float] | None = None,
                 active_models: List[str] = None,
                 top_k_models: int = 3,
                 use_soft_fusion: bool = True,
                 corr_penalty_enabled: bool = False,
                 corr_penalty_scale: float = 0.2):
        self.model_cols = model_cols
        self.bucket_best_models = {}
        self.bucket_weights = {}
        self.min_bucket_size = min_bucket_size
        self.high_load_thr = None
        self.global_weights = {}
        self.global_models = []
        self.is_daily = False
        self.volatility_col = None
        self.volatility_thr = None
        self.robust_cfg = robust_cfg or ROBUST_CFG
        self.active_models = active_models
        self.top_k_models = top_k_models
        self.use_soft_fusion = use_soft_fusion
        self._effective_models = None
        self.corr_penalty_enabled = corr_penalty_enabled
        self.corr_penalty_scale = corr_penalty_scale
        self._corr_penalty_factors: Dict[str, float] = {}

    def _detect_daily(self, df: pd.DataFrame) -> bool:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
            if len(ts) >= 2:
                deltas = ts.sort_values().diff().dropna()
                if not deltas.empty:
                    return deltas.median() >= pd.Timedelta(hours=23)
        if "ctx_hour" in df.columns:
            hours = df["ctx_hour"].dropna().unique()
            if len(hours) <= 1:
                return True
        return False

    def _pick_volatility_col(self, df: pd.DataFrame) -> Optional[str]:
        candidates = [c for c in df.columns if c.startswith("ctx_roll") and "std" in c]
        if not candidates:
            candidates = [c for c in df.columns if c.startswith("ctx_roll") and ("var" in c or "iqr" in c)]
        return candidates[0] if candidates else None

    def _compute_corr_penalty_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        if "y" not in df.columns or not self._effective_models:
            return {}
        y = df["y"].values
        n_models = len(self._effective_models)
        if n_models <= 1:
            return {m: 1.0 for m in self._effective_models}
        errors = []
        for m in self._effective_models:
            if m not in df.columns:
                return {}
            errors.append(df[m].values - y)
        err_mat = np.column_stack(errors)
        corr = np.corrcoef(err_mat.T)
        corr = np.nan_to_num(corr, nan=1.0, posinf=1.0, neginf=1.0)
        mean_abs_corr = []
        for i in range(n_models):
            vals = [abs(float(corr[i, j])) for j in range(n_models) if j != i]
            mean_abs_corr.append(float(np.mean(vals)) if vals else 0.0)
        penalty_factor = np.exp(-float(self.corr_penalty_scale) * np.asarray(mean_abs_corr))
        return {m: float(penalty_factor[i]) for i, m in enumerate(self._effective_models)}

    def _append_part(self, bucket: np.ndarray, part: np.ndarray) -> np.ndarray:
        mask = part != ""
        return np.where(mask & (bucket != ""), bucket + "_" + part, np.where(mask, part, bucket))

    def _assign_buckets(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        bucket = np.array([""] * n, dtype=object)

        if self.is_daily:
            if "ctx_is_weekend" in df.columns:
                weekend = df["ctx_is_weekend"].fillna(0).astype(int).values
                part = np.where(weekend == 1, "weekend", "weekday")
                bucket = self._append_part(bucket, part)
            if "ctx_is_holiday" in df.columns:
                holiday = df["ctx_is_holiday"].fillna(0).astype(int).values
                part = np.where(holiday == 1, "holiday", "")
                bucket = self._append_part(bucket, part)
            if self.volatility_thr and self.volatility_col and self.volatility_col in df.columns:
                v = pd.to_numeric(df[self.volatility_col], errors="coerce").values
                low, high = self.volatility_thr
                part = np.where(
                    np.isnan(v),
                    "",
                    np.where(v <= low, "vol_low", np.where(v <= high, "vol_mid", "vol_high"))
                )
                bucket = self._append_part(bucket, part)
        else:
            if "ctx_hour" in df.columns:
                hour = pd.to_numeric(df["ctx_hour"], errors="coerce").fillna(12).astype(int).values
                part = np.where(
                    (hour >= 7) & (hour < 10),
                    "morning_peak",
                    np.where(
                        (hour >= 10) & (hour < 17),
                        "daytime",
                        np.where((hour >= 17) & (hour < 21), "evening_peak", "night"),
                    ),
                )
                bucket = self._append_part(bucket, part)
            if "ctx_is_weekend" in df.columns:
                weekend = df["ctx_is_weekend"].fillna(0).astype(int).values
                part = np.where(weekend == 1, "weekend", "weekday")
                bucket = self._append_part(bucket, part)
            if "ctx_is_holiday" in df.columns:
                holiday = df["ctx_is_holiday"].fillna(0).astype(int).values
                part = np.where(holiday == 1, "holiday", "")
                bucket = self._append_part(bucket, part)
            if self.high_load_thr is not None and "y" in df.columns:
                y = pd.to_numeric(df["y"], errors="coerce").values
                part = np.where(np.abs(y) >= self.high_load_thr, "highload", "")
                bucket = self._append_part(bucket, part)

        bucket = np.where(bucket == "", "default", bucket)
        return pd.Series(bucket, index=df.index)

    def _get_bucket_id(self, row: pd.Series) -> str:
        parts = []

        if self.is_daily:
            if "ctx_is_weekend" in row.index:
                parts.append("weekend" if row["ctx_is_weekend"] else "weekday")
            if "ctx_is_holiday" in row.index and row["ctx_is_holiday"]:
                parts.append("holiday")
            if self.volatility_thr and self.volatility_col and self.volatility_col in row.index:
                v = row[self.volatility_col]
                if pd.notna(v):
                    low, high = self.volatility_thr
                    if v <= low:
                        parts.append("vol_low")
                    elif v <= high:
                        parts.append("vol_mid")
                    else:
                        parts.append("vol_high")
        else:
            if "ctx_hour" in row.index:
                hour = int(row["ctx_hour"]) if pd.notna(row["ctx_hour"]) else 12
                if 7 <= hour < 10:
                    parts.append("morning_peak")
                elif 10 <= hour < 17:
                    parts.append("daytime")
                elif 17 <= hour < 21:
                    parts.append("evening_peak")
                else:
                    parts.append("night")

            if "ctx_is_weekend" in row.index:
                parts.append("weekend" if row["ctx_is_weekend"] else "weekday")

            if "ctx_is_holiday" in row.index and row["ctx_is_holiday"]:
                parts.append("holiday")

            if self.high_load_thr is not None and "y" in row.index:
                if abs(float(row["y"])) >= self.high_load_thr:
                    parts.append("highload")
        return "_".join(parts) if parts else "default"

    def fit(self, val_df: pd.DataFrame):
        self.is_daily = self._detect_daily(val_df)
        self.volatility_col = self._pick_volatility_col(val_df)
        if self.volatility_col:
            vals = pd.to_numeric(val_df[self.volatility_col], errors="coerce").dropna()
            if len(vals) >= self.min_bucket_size:
                low = float(np.percentile(vals, 33))
                high = float(np.percentile(vals, 67))
                self.volatility_thr = (low, high)
            else:
                self.volatility_thr = None

        if self.active_models is not None:
            self._effective_models = [m for m in self.active_models if m in self.model_cols and m in val_df.columns]
            if not self._effective_models:
                self._effective_models = [m for m in self.model_cols if m in val_df.columns]
        else:
            self._effective_models = [m for m in self.model_cols if m in val_df.columns]

        if len(val_df) > 0:
            q = self.robust_cfg.get("high_load_q", 0.90)
            self.high_load_thr = float(np.percentile(np.abs(val_df["y"].values), q * 100))
        else:
            self.high_load_thr = None

        val_df = val_df.copy()
        val_df["_bucket"] = self._assign_buckets(val_df)

        for bucket_id, group in val_df.groupby("_bucket"):
            if len(group) < self.min_bucket_size:
                continue

            y_bucket = group["y"].values

            model_maes = {}
            for m in self._effective_models:
                if m in group.columns:
                    if self.robust_cfg.get("enable", False):
                        score = robust_mae(y_bucket, group[m].values, self.robust_cfg)
                    else:
                        score = float(mean_absolute_error(y_bucket, group[m].values))
                    model_maes[m] = score

            if not model_maes:
                continue

            sorted_models = sorted(model_maes.items(), key=lambda x: x[1])
            best_models = [m for m, _ in sorted_models[:self.top_k_models]]
            self.bucket_best_models[bucket_id] = best_models

            if self.use_soft_fusion and len(best_models) > 1:
                inv_maes = {m: 1.0 / (model_maes[m] + 1e-6) for m in best_models}
                if self.corr_penalty_enabled:
                    inv_maes = {
                        m: inv_maes[m] * float(self._corr_penalty_factors.get(m, 1.0))
                        for m in inv_maes
                    }
                inv_sum = sum(inv_maes.values())
                normalized = {m: inv_maes[m] / inv_sum for m in best_models}
                self.bucket_weights[bucket_id] = normalized
            else:
                X_bucket = group[best_models].values
                reg = Ridge(alpha=1.0, fit_intercept=False)
                sample_weight = None
                if self.robust_cfg.get("enable", False):
                    base_pred = np.mean(X_bucket, axis=1)
                    abs_err = np.abs(y_bucket - base_pred)
                    sample_weight = compute_extreme_weights(y_bucket, abs_err, self.robust_cfg)
                reg.fit(X_bucket, y_bucket, sample_weight=sample_weight)

                raw_weights = reg.coef_
                clipped = np.maximum(raw_weights, 0)
                if self.corr_penalty_enabled:
                    penalty = np.array(
                        [float(self._corr_penalty_factors.get(m, 1.0)) for m in best_models],
                        dtype=float,
                    )
                    clipped = clipped * penalty
                weight_sum = clipped.sum()
                if weight_sum > 0:
                    normalized = dict(zip(best_models, clipped / weight_sum))
                else:
                    normalized = {m: 1.0 / len(best_models) for m in best_models}
                self.bucket_weights[bucket_id] = normalized

        print(f"    [BucketSelector] 学习了 {len(self.bucket_best_models)} 个场景桶 (min_size={self.min_bucket_size}, top_k={self.top_k_models})")

        if self._effective_models:
            y_all = val_df["y"].values
            if self.robust_cfg.get("enable", False):
                model_maes = {m: robust_mae(y_all, val_df[m].values, self.robust_cfg) for m in self._effective_models}
            else:
                model_maes = {m: float(mean_absolute_error(y_all, val_df[m].values)) for m in self._effective_models}
            sorted_models = sorted(model_maes.items(), key=lambda x: x[1])
            best_models = [m for m, _ in sorted_models[:self.top_k_models]]

            if self.use_soft_fusion and len(best_models) > 1:
                inv_maes = {m: 1.0 / (model_maes[m] + 1e-6) for m in best_models}
                if self.corr_penalty_enabled:
                    inv_maes = {
                        m: inv_maes[m] * float(self._corr_penalty_factors.get(m, 1.0))
                        for m in inv_maes
                    }
                inv_sum = sum(inv_maes.values())
                self.global_weights = {m: inv_maes[m] / inv_sum for m in best_models}
            else:
                X_all = val_df[best_models].values
                reg = Ridge(alpha=1.0, fit_intercept=False)
                sample_weight = None
                if self.robust_cfg.get("enable", False):
                    base_pred = np.mean(X_all, axis=1)
                    abs_err = np.abs(y_all - base_pred)
                    sample_weight = compute_extreme_weights(y_all, abs_err, self.robust_cfg)
                reg.fit(X_all, y_all, sample_weight=sample_weight)
                raw_weights = reg.coef_
                clipped = np.maximum(raw_weights, 0)
                if self.corr_penalty_enabled:
                    penalty = np.array(
                        [float(self._corr_penalty_factors.get(m, 1.0)) for m in best_models],
                        dtype=float,
                    )
                    clipped = clipped * penalty
                weight_sum = clipped.sum()
                if weight_sum > 0:
                    normalized = clipped / weight_sum
                else:
                    normalized = np.ones(len(best_models)) / len(best_models)
                self.global_weights = dict(zip(best_models, normalized))
            self.global_models = best_models

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        test_df = test_df.copy()
        test_df["_bucket"] = self._assign_buckets(test_df)

        predictions = np.zeros(len(test_df))
        models_used_per_sample = np.zeros(len(test_df))

        def _apply_fallback(mask: np.ndarray):
            if self.global_weights:
                models = [m for m in self.global_weights.keys() if m in test_df.columns]
                if models:
                    w = np.array([self.global_weights[m] for m in models])
                    predictions[mask] = test_df.loc[mask, models].values @ w
                    models_used_per_sample[mask] = len(models)
                    return
            models = [m for m in self.model_cols if m in test_df.columns]
            if models:
                predictions[mask] = test_df.loc[mask, models].values.mean(axis=1)
                models_used_per_sample[mask] = len(models)

        for bucket_id in test_df["_bucket"].unique():
            mask = test_df["_bucket"].values == bucket_id
            if bucket_id in self.bucket_weights:
                weights = self.bucket_weights[bucket_id]
                models = [m for m in weights.keys() if m in test_df.columns]
                if models:
                    w = np.array([weights[m] for m in models])
                    predictions[mask] = test_df.loc[mask, models].values @ w
                    models_used_per_sample[mask] = len(models)
                else:
                    _apply_fallback(mask)
            else:
                _apply_fallback(mask)

        avg_models_used = float(models_used_per_sample.mean()) if len(models_used_per_sample) > 0 else len(self.model_cols)
        return predictions, avg_models_used
