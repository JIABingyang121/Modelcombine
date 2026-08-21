from __future__ import annotations
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import warnings

# [Fix] 不再全局屏蔽警告，改为在特定位置局部屏蔽
# warnings.filterwarnings('ignore')  # 已移除

# Optional backends
try:
    from xgboost import XGBRegressor as _XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    _XGBRegressor = None

try:
    from lightgbm import LGBMRegressor as _LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    _LGBMRegressor = None

try:
    from catboost import CatBoostRegressor as _CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    _CatBoostRegressor = None

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency
    Prophet = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover - optional dependency
    ARIMA = None
    seasonal_decompose = None
    ExponentialSmoothing = None

try:
    import pmdarima as pm
except Exception:  # pragma: no cover - optional dependency
    pm = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no", ""}


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _resolve_gpu_id() -> Optional[int]:
    raw = os.environ.get("MODELCOMBINE_GPU_ID")
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw.strip())
    except Exception:
        return None


def _looks_like_gpu_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("gpu", "cuda", "nvidia", "device", "opencl", "cudart"))


def _count_convergence_warnings(ws: List[warnings.WarningMessage]) -> int:
    cnt = 0
    for w in ws:
        cname = getattr(getattr(w, "category", None), "__name__", "")
        if "ConvergenceWarning" in str(cname):
            cnt += 1
    return cnt


def _flatten_estimators(estimators: Any) -> List[Any]:
    if estimators is None:
        return []
    arr = np.asarray(estimators, dtype=object).reshape(-1)
    return [est for est in arr.tolist() if hasattr(est, "predict")]


def _quantile_head_predict_interval(
    wrapper: Any,
    X: pd.DataFrame,
    alpha: float = 0.1,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """用原生分位数目标训练上下界模型（惰性、按 alpha 缓存）；不可用时返回 None。"""
    X_train = getattr(wrapper, "_train_X", None)
    y_train = getattr(wrapper, "_train_y", None)
    build = getattr(wrapper, "_build_quantile_model", None)
    if X_train is None or y_train is None or build is None:
        return None
    cache = getattr(wrapper, "_quantile_models", None)
    if cache is None:
        cache = {}
        wrapper._quantile_models = cache
    key = round(float(alpha), 6)
    try:
        if key not in cache:
            lo_model = build(alpha / 2.0)
            hi_model = build(1.0 - alpha / 2.0)
            if lo_model is None or hi_model is None:
                return None
            lo_model.fit(X_train, y_train)
            hi_model.fit(X_train, y_train)
            cache[key] = (lo_model, hi_model)
        lo_model, hi_model = cache[key]
        yhat = np.asarray(wrapper.model.predict(X), dtype=float).reshape(-1)
        lower = np.asarray(lo_model.predict(X), dtype=float).reshape(-1)
        upper = np.asarray(hi_model.predict(X), dtype=float).reshape(-1)
    except Exception:
        return None
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    lower = np.minimum(lower, yhat)
    upper = np.maximum(upper, yhat)
    return yhat, lower, upper


def _training_residual_predict_interval(
    wrapper: Any,
    X: pd.DataFrame,
    alpha: float = 0.1,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """基于训练残差经验分位数构造区间（近似，残差为样本内）；不可用时返回 None。"""
    X_train = getattr(wrapper, "_train_X", None)
    y_train = getattr(wrapper, "_train_y", None)
    if X_train is None or y_train is None:
        return None
    try:
        yhat = np.asarray(wrapper.model.predict(X), dtype=float).reshape(-1)
        train_pred = np.asarray(wrapper.model.predict(X_train), dtype=float).reshape(-1)
        residuals = np.asarray(y_train, dtype=float).reshape(-1) - train_pred
    except Exception:
        return None
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 10:
        return None
    lo, hi = np.quantile(residuals, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower = np.minimum(yhat + lo, yhat)
    upper = np.maximum(yhat + hi, yhat)
    return yhat, lower, upper


def _tree_ensemble_predict_interval(
    model: Any,
    X: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yhat = np.asarray(model.predict(X), dtype=float).reshape(-1)
    estimators = _flatten_estimators(getattr(model, "estimators_", None))
    if not estimators:
        return yhat, yhat.copy(), yhat.copy()

    preds = []
    for estimator in estimators:
        try:
            pred = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
        except Exception:
            continue
        if len(pred) == len(yhat):
            preds.append(pred)
    if len(preds) < 2:
        return yhat, yhat.copy(), yhat.copy()

    stacked = np.vstack(preds)
    lower = np.quantile(stacked, alpha / 2.0, axis=0)
    upper = np.quantile(stacked, 1.0 - alpha / 2.0, axis=0)
    lower = np.minimum(lower, yhat)
    upper = np.maximum(upper, yhat)
    return yhat, lower, upper


class XGBModel(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self._gpu_requested = False
        if _XGBRegressor is not None:
            xgb_params = dict(params)
            tree_n_jobs = _env_int("MODELCOMBINE_TREE_N_JOBS", 0)
            if tree_n_jobs > 0 and "n_jobs" not in xgb_params:
                xgb_params["n_jobs"] = tree_n_jobs

            self._gpu_requested = _env_flag("MODELCOMBINE_USE_GPU", True)
            if self._gpu_requested:
                gpu_id = _resolve_gpu_id()
                # 避免同时设置 device 和 gpu_id（XGBoost 会报冲突并回退 CPU）。
                xgb_params.setdefault("tree_method", "hist")
                if "device" not in xgb_params and "gpu_id" not in xgb_params:
                    xgb_params["device"] = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
                # predictor 在新版本中会被忽略，且会产生警告，默认不显式设置。
                xgb_params.pop("predictor", None)

            self._params = xgb_params
            self.model = _XGBRegressor(**xgb_params)
        else:
            print("[xgboost_reg] WARNING: xgboost not available, falling back to GradientBoostingRegressor")
            # Fallback: approximate with GradientBoosting
            self.model = GradientBoostingRegressor(**{k: v for k, v in params.items() if k in {"n_estimators", "learning_rate", "max_depth"}})
            self._params = dict(params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            self.model.fit(X, y)
        except Exception as exc:
            can_fallback = (
                _XGBRegressor is not None
                and self._gpu_requested
                and _env_flag("MODELCOMBINE_GPU_FALLBACK_CPU", True)
                and _looks_like_gpu_error(exc)
            )
            if not can_fallback:
                raise
            print(f"[xgboost_reg] GPU fit failed, fallback to CPU: {exc}")
            cpu_params = dict(self._params)
            cpu_params.pop("device", None)
            cpu_params.pop("gpu_id", None)
            cpu_params.pop("predictor", None)
            cpu_params["tree_method"] = "hist"
            self.model = _XGBRegressor(**cpu_params)
            self.model.fit(X, y)
        self._train_X, self._train_y = X, y
        self._quantile_models = {}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def _build_quantile_model(self, q: float):
        if _XGBRegressor is None:
            return None
        params = dict(self._params)
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = float(q)
        return _XGBRegressor(**params)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = _quantile_head_predict_interval(self, X, alpha=alpha)
        if result is not None:
            return result
        result = _training_residual_predict_interval(self, X, alpha=alpha)
        if result is not None:
            return result
        return _tree_ensemble_predict_interval(self.model, X, alpha=alpha)


class LGBMModel(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self._gpu_requested = False
        if _LGBMRegressor is not None:
            lgbm_params = dict(params)
            tree_n_jobs = _env_int("MODELCOMBINE_TREE_N_JOBS", 0)
            if tree_n_jobs > 0 and "n_jobs" not in lgbm_params:
                lgbm_params["n_jobs"] = tree_n_jobs

            self._gpu_requested = _env_flag("MODELCOMBINE_USE_GPU", True)
            if self._gpu_requested:
                gpu_id = _resolve_gpu_id()
                if "device_type" not in lgbm_params and "device" not in lgbm_params:
                    lgbm_params["device_type"] = "gpu"
                if gpu_id is not None and "gpu_device_id" not in lgbm_params:
                    lgbm_params["gpu_device_id"] = gpu_id
                lgbm_params.setdefault("max_bin", 255)

            self._params = lgbm_params
            self.model = _LGBMRegressor(**lgbm_params)
        else:
            # Fallback: approximate with RandomForest
            self.model = RandomForestRegressor(**{k: v for k, v in params.items() if k in {"n_estimators", "max_depth", "random_state"}})
            self._params = dict(params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            self.model.fit(X, y)
        except Exception as exc:
            can_fallback = (
                _LGBMRegressor is not None
                and self._gpu_requested
                and _env_flag("MODELCOMBINE_GPU_FALLBACK_CPU", True)
                and _looks_like_gpu_error(exc)
            )
            if not can_fallback:
                raise
            print(f"[lgbm_reg] GPU fit failed, fallback to CPU: {exc}")
            cpu_params = dict(self._params)
            cpu_params.pop("gpu_device_id", None)
            cpu_params.pop("device", None)
            cpu_params["device_type"] = "cpu"
            self.model = _LGBMRegressor(**cpu_params)
            self.model.fit(X, y)
        self._train_X, self._train_y = X, y
        self._quantile_models = {}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def _build_quantile_model(self, q: float):
        if _LGBMRegressor is None:
            return None
        params = dict(self._params)
        params["objective"] = "quantile"
        params["alpha"] = float(q)
        return _LGBMRegressor(**params)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = _quantile_head_predict_interval(self, X, alpha=alpha)
        if result is not None:
            return result
        result = _training_residual_predict_interval(self, X, alpha=alpha)
        if result is not None:
            return result
        return _tree_ensemble_predict_interval(self.model, X, alpha=alpha)


class CatBoostModel(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self._gpu_requested = False
        params = {"verbose": False, **params}
        if _CatBoostRegressor is not None:
            cat_params = dict(params)
            tree_n_jobs = _env_int("MODELCOMBINE_TREE_N_JOBS", 0)
            if tree_n_jobs > 0 and "thread_count" not in cat_params:
                cat_params["thread_count"] = tree_n_jobs

            force_cpu = _env_flag("MODELCOMBINE_CATBOOST_FORCE_CPU", True)
            self._gpu_requested = _env_flag("MODELCOMBINE_USE_GPU", True) and (not force_cpu)
            if force_cpu:
                cat_params["task_type"] = "CPU"
                cat_params.pop("devices", None)
            elif self._gpu_requested:
                gpu_id = _resolve_gpu_id()
                cat_params.setdefault("task_type", "GPU")
                if gpu_id is not None and "devices" not in cat_params:
                    cat_params["devices"] = str(gpu_id)

            self._params = cat_params
            self.model = _CatBoostRegressor(**cat_params)
        else:
            # Fallback: approximate with GradientBoosting (与 XGB fallback 做参数区分，降低同质化)
            _fb_params = {k: v for k, v in params.items() if k in {"n_estimators", "learning_rate", "max_depth"}}
            _fb_params.setdefault("max_depth", 5)
            _fb_params.setdefault("subsample", 0.8)
            _fb_params.setdefault("min_samples_leaf", 10)
            print("[catboost_reg] WARNING: catboost not available, falling back to GradientBoostingRegressor(max_depth=5)")
            self.model = GradientBoostingRegressor(**_fb_params)
            self._params = dict(params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            self.model.fit(X, y)
        except Exception as exc:
            can_fallback = (
                _CatBoostRegressor is not None
                and self._gpu_requested
                and _env_flag("MODELCOMBINE_GPU_FALLBACK_CPU", True)
                and _looks_like_gpu_error(exc)
            )
            if not can_fallback:
                raise
            print(f"[catboost_reg] GPU fit failed, fallback to CPU: {exc}")
            cpu_params = dict(self._params)
            cpu_params["task_type"] = "CPU"
            cpu_params.pop("devices", None)
            self.model = _CatBoostRegressor(**cpu_params)
            self.model.fit(X, y)
        self._train_X, self._train_y = X, y
        self._quantile_models = {}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def _build_quantile_model(self, q: float):
        if _CatBoostRegressor is None:
            return None
        params = dict(self._params)
        # Quantile 损失仅支持 CPU，避免 GPU 参数组合报错
        params["task_type"] = "CPU"
        params.pop("devices", None)
        params["loss_function"] = f"Quantile:alpha={float(q)}"
        return _CatBoostRegressor(**params)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = _quantile_head_predict_interval(self, X, alpha=alpha)
        if result is not None:
            return result
        result = _training_residual_predict_interval(self, X, alpha=alpha)
        if result is not None:
            return result
        return _tree_ensemble_predict_interval(self.model, X, alpha=alpha)


# 电力需求预测专用模型

class ProphetModel(BaseEstimator, RegressorMixin):
    """Prophet时序预测模型，适合处理季节性和节假日效应"""
    def __init__(self, **params):
        self.params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True, 
            'daily_seasonality': True,
            'seasonality_mode': 'multiplicative',
            **params
        }
        self.model = None
        self.feature_cols: List[str] = []
        self.start_date = '2023-01-01'
        self.freq: str | None = None
        self.train_end = None
        self.fit_status: Dict[str, Any] = {
            "fit_ok": False,
            "model_family": "prophet",
            "fallback_used": False,
            "convergence_warning_count": 0,
            "warning_messages": [],
        }
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if Prophet is None:
            raise ImportError("Prophet not available. Install with: pip install prophet")
        # 仅使用目标序列，忽略外部回归以避免列不一致
        self.feature_cols = []

        # 使用真实时间索引来训练，避免起始时间重置导致的错位
        if isinstance(y.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(y.index)
            self.freq = inferred if inferred is not None else 'H'
            ds = pd.to_datetime(y.index)
        else:
            self.freq = 'H'
            ds = pd.date_range(self.start_date, periods=len(y), freq=self.freq)

        df = pd.DataFrame({'ds': ds, 'y': y.values})

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            self.model = Prophet(**self.params)
            self.model.fit(df)
        conv_cnt = _count_convergence_warnings(ws)
        warning_messages = [str(w.message)[:300] for w in ws[:8]]
        self.fit_status = {
            "fit_ok": True,
            "model_family": "prophet",
            "fallback_used": False,
            "convergence_warning_count": int(conv_cnt),
            "warning_messages": warning_messages,
        }
        self.train_end = df['ds'].iloc[-1]
        return self

    def _make_future_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        # 使用评估集的真实时间戳进行预测，确保时间对齐；如缺失则从训练末尾顺延
        if hasattr(X, "index") and isinstance(X.index, pd.DatetimeIndex) and len(X.index) > 0:
            ds_future = pd.to_datetime(X.index)
        else:
            freq = self.freq or 'H'
            start = self.train_end if self.train_end is not None else pd.to_datetime(self.start_date)
            ds_future = pd.date_range(start=start + pd.tseries.frequencies.to_offset(freq), periods=len(X), freq=freq)
        return pd.DataFrame({'ds': ds_future})

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted yet")

        future = self._make_future_frame(X)
        # [Fix] 局部屏蔽 Prophet predict 的警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            forecast = self.model.predict(future)
        return forecast['yhat'].values

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not fitted yet")

        future = self._make_future_frame(X)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            forecast = self.model.predict(future)
        yhat = forecast['yhat'].values
        lower = forecast.get('yhat_lower', forecast['yhat']).values
        upper = forecast.get('yhat_upper', forecast['yhat']).values
        return yhat, lower, upper

    def get_fit_status(self) -> Dict[str, Any]:
        return dict(self.fit_status)

class ARIMAModel(BaseEstimator, RegressorMixin):
    """Auto-ARIMA + ETS 回退的时序预测模型"""
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), freq: str = 'H', **params):
        self.order = order
        self.seasonal_order = seasonal_order
        self.freq = freq
        self.params = params
        self.fitted_model = None
        self.start_date = '2023-01-01'
        self._model_family = None
        self._seasonal_period = 24
        self.fit_status: Dict[str, Any] = {
            "fit_ok": False,
            "model_family": None,
            "fallback_used": False,
            "convergence_warning_count": 0,
            "warning_messages": [],
            "attempts": [],
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_values = np.asarray(y.values, dtype=float)
        if len(y_values) == 0:
            raise ValueError("Empty target for ARIMAModel")

        # 频率到季节周期的保守映射。
        freq_l = str(self.freq).lower()
        if "h" in freq_l:
            self._seasonal_period = 24
        elif "d" in freq_l:
            self._seasonal_period = 7
        else:
            self._seasonal_period = 24

        use_auto_arima = _env_flag("MODELCOMBINE_USE_AUTO_ARIMA", True)
        if use_auto_arima and pm is not None:
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    self.fitted_model = pm.auto_arima(
                        y_values,
                        seasonal=True,
                        m=max(2, int(self._seasonal_period)),
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                        max_p=5,
                        max_q=5,
                        max_P=2,
                        max_Q=2,
                        max_order=10,
                        n_fits=50,
                    )
                self._model_family = "auto_arima"
                conv_cnt = _count_convergence_warnings(ws)
                warning_messages = [str(w.message)[:300] for w in ws[:8]]
                self.fit_status = {
                    "fit_ok": True,
                    "model_family": "auto_arima",
                    "fallback_used": False,
                    "convergence_warning_count": int(conv_cnt),
                    "warning_messages": warning_messages,
                    "attempts": [{"family": "auto_arima", "ok": True}],
                }
                return self
            except Exception as exc:
                self.fit_status["attempts"].append({
                    "family": "auto_arima",
                    "ok": False,
                    "error": str(exc),
                })
                print(f"[arima] AutoARIMA failed, fallback to ETS/ARIMA: {exc}")

        if ExponentialSmoothing is not None:
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    self.fitted_model = ExponentialSmoothing(
                        y_values,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=max(2, int(self._seasonal_period)),
                        damped_trend=True,
                    ).fit(optimized=True)
                self._model_family = "ets"
                conv_cnt = _count_convergence_warnings(ws)
                warning_messages = [str(w.message)[:300] for w in ws[:8]]
                self.fit_status = {
                    "fit_ok": True,
                    "model_family": "ets",
                    "fallback_used": True,
                    "convergence_warning_count": int(conv_cnt),
                    "warning_messages": warning_messages,
                    "attempts": self.fit_status.get("attempts", []) + [{"family": "ets", "ok": True}],
                }
                return self
            except Exception as exc:
                self.fit_status["attempts"].append({
                    "family": "ets",
                    "ok": False,
                    "error": str(exc),
                })
                print(f"[arima] ETS failed, fallback to ARIMA: {exc}")

        if ARIMA is None:
            raise ImportError("statsmodels/pmdarima not available for ARIMA/ETS fallback")

        if isinstance(y.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(y.index)
            if inferred is None:
                y_index = pd.RangeIndex(start=0, stop=len(y))
            else:
                y_index = y.index
        else:
            y_index = pd.date_range(self.start_date, periods=len(y), freq=self.freq)
        y_series = pd.Series(y.values, index=y_index)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            self.fitted_model = ARIMA(y_series, order=self.order, seasonal_order=self.seasonal_order, **self.params).fit()
        self._model_family = "arima"
        conv_cnt = _count_convergence_warnings(ws)
        warning_messages = [str(w.message)[:300] for w in ws[:8]]
        self.fit_status = {
            "fit_ok": True,
            "model_family": "arima",
            "fallback_used": True,
            "convergence_warning_count": int(conv_cnt),
            "warning_messages": warning_messages,
            "attempts": self.fit_status.get("attempts", []) + [{"family": "arima", "ok": True}],
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")

        steps = len(X)
        if self._model_family == "auto_arima":
            forecast = self.fitted_model.predict(n_periods=steps)
            return np.asarray(forecast, dtype=float)
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values if hasattr(forecast, 'values') else np.asarray(forecast, dtype=float)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")

        steps = len(X)
        if self._model_family == "auto_arima":
            forecast = self.fitted_model.predict(
                n_periods=steps,
                return_conf_int=True,
                alpha=alpha,
            )
            yhat, conf_int = forecast
            conf = np.asarray(conf_int, dtype=float)
            return (
                np.asarray(yhat, dtype=float),
                conf[:, 0],
                conf[:, 1],
            )

        if hasattr(self.fitted_model, "get_forecast"):
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            yhat = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=alpha)
            conf = np.asarray(conf_int, dtype=float)
            return (
                np.asarray(yhat, dtype=float),
                conf[:, 0],
                conf[:, 1],
            )

        yhat = self.predict(X)
        return yhat, yhat.copy(), yhat.copy()

    def get_fit_status(self) -> Dict[str, Any]:
        return dict(self.fit_status)


class SeasonalNaiveModel(BaseEstimator, RegressorMixin):
    """Seasonal Naive 基线模型：使用上一个季节周期的观测值进行预测"""
    def __init__(self, seasonal_period: int = 24):
        self.seasonal_period = int(seasonal_period)
        self.season_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if y is None or len(y) == 0:
            raise ValueError("Empty target series for SeasonalNaiveModel")

        period = max(1, self.seasonal_period)
        y_values = np.asarray(y)
        if len(y_values) >= period:
            self.season_ = y_values[-period:]
        else:
            self.season_ = y_values.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.season_ is None:
            raise ValueError("Model not fitted yet")

        steps = len(X)
        if steps <= 0:
            return np.asarray([])

        if len(self.season_) == 1:
            return np.full(steps, self.season_[0], dtype=float)

        reps = int(np.ceil(steps / len(self.season_)))
        forecast = np.tile(self.season_, reps)[:steps]
        return forecast.astype(float)
 
class PowerLoadDifferenceModel(BaseEstimator, RegressorMixin):
    """电力负荷差异分析模型，专门处理不同区域间的用电差异"""
    def __init__(self, base_model='ridge', alpha=1.0, **params):
        self.base_model_type = base_model
        self.alpha = alpha
        self.params = params
        self.base_model = None
        self.region_scalers = {}
        self.region_stats = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # 初始化基础模型
        if self.base_model_type == 'ridge':
            self.base_model = Ridge(alpha=self.alpha, **self.params)
        elif self.base_model_type == 'linear':
            self.base_model = LinearRegression(**self.params)
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_type}")
            
        # 计算区域统计信息
        if 'region' in X.columns:
            for region in X['region'].unique():
                mask = X['region'] == region
                region_data = y[mask]
                self.region_stats[region] = {
                    'mean': region_data.mean(),
                    'std': region_data.std(),
                    'peak_hour': X[mask].groupby('hour')['load'].mean().idxmax() if 'hour' in X.columns else 12
                }
                
                # 为每个区域创建标准化器
                scaler = StandardScaler()
                self.region_scalers[region] = scaler.fit(region_data.values.reshape(-1, 1))
        
        # 创建差异特征
        X_diff = self._create_difference_features(X, y)
        
        # 训练基础模型
        self.base_model.fit(X_diff, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.base_model is None:
            raise ValueError("Model not fitted yet")
            
        X_diff = self._create_difference_features(X)
        base_pred = self.base_model.predict(X_diff)
        
        # 应用区域特定的调整
        adjusted_pred = base_pred.copy()
        if 'region' in X.columns:
            for i, region in enumerate(X['region']):
                if region in self.region_stats:
                    stats = self.region_stats[region]
                    # 根据区域特征调整预测
                    hour = X.iloc[i]['hour'] if 'hour' in X.columns else 12
                    peak_adjustment = 1.2 if abs(hour - stats['peak_hour']) <= 2 else 1.0
                    adjusted_pred[i] *= peak_adjustment
                    
        return adjusted_pred
        
    def _create_difference_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """创建差异特征"""
        X_diff = X.copy()
        
        # 添加时间差异特征
        if 'hour' in X.columns:
            X_diff['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
            X_diff['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
            
        if 'day_of_week' in X.columns:
            X_diff['dow_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
            X_diff['dow_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
            
        # 添加区域间差异特征
        if 'region' in X.columns and len(self.region_stats) > 0:
            for region in X['region'].unique():
                if region in self.region_stats:
                    mask = X['region'] == region
                    stats = self.region_stats[region]
                    X_diff.loc[mask, f'{region}_mean_diff'] = stats['mean']
                    X_diff.loc[mask, f'{region}_std_ratio'] = stats['std']
                    
        # 移除非数值列
        numeric_cols = X_diff.select_dtypes(include=[np.number]).columns
        return X_diff[numeric_cols]


class MultiModalFusionModel(BaseEstimator, RegressorMixin):
    """多模态数据融合模型，整合用电、天气、时间等多种数据源"""
    def __init__(self, fusion_method='attention', **params):
        self.fusion_method = fusion_method
        self.params = params
        self.load_model = Ridge(alpha=1.0)
        self.weather_model = Ridge(alpha=1.0) 
        self.time_model = Ridge(alpha=1.0)
        self.fusion_model = Ridge(alpha=1.0)
        self.feature_groups = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # 分组特征
        self._group_features(X)
        
        # 分别训练各模态模型
        if self.feature_groups.get('load_features'):
            self.load_model.fit(X[self.feature_groups['load_features']], y)
            
        if self.feature_groups.get('weather_features'):
            self.weather_model.fit(X[self.feature_groups['weather_features']], y)
            
        if self.feature_groups.get('time_features'):
            self.time_model.fit(X[self.feature_groups['time_features']], y)
            
        # 创建融合特征
        fusion_features = self._create_fusion_features(X, y)
        self.fusion_model.fit(fusion_features, y)
        
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        fusion_features = self._create_fusion_features(X)
        return self.fusion_model.predict(fusion_features)
        
    def _group_features(self, X: pd.DataFrame):
        """将特征分组到不同模态"""
        load_cols = [col for col in X.columns if any(keyword in col.lower() 
                    for keyword in ['load', 'power', 'consumption', 'demand', 'lag'])]
        weather_cols = [col for col in X.columns if any(keyword in col.lower() 
                       for keyword in ['temp', 'humidity', 'wind', 'weather', 'rain'])]
        time_cols = [col for col in X.columns if any(keyword in col.lower() 
                    for keyword in ['hour', 'day', 'week', 'month', 'holiday', 'season'])]
        
        self.feature_groups = {
            'load_features': load_cols,
            'weather_features': weather_cols, 
            'time_features': time_cols
        }
        
    def _create_fusion_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """创建融合特征"""
        fusion_features = []
        
        # 各模态的预测作为融合特征
        if 'load_features' in self.feature_groups and self.feature_groups['load_features']:
            load_pred = self.load_model.predict(X[self.feature_groups['load_features']])
            fusion_features.append(load_pred.reshape(-1, 1))
            
        if 'weather_features' in self.feature_groups and self.feature_groups['weather_features']:
            weather_pred = self.weather_model.predict(X[self.feature_groups['weather_features']])
            fusion_features.append(weather_pred.reshape(-1, 1))
            
        if 'time_features' in self.feature_groups and self.feature_groups['time_features']:
            time_pred = self.time_model.predict(X[self.feature_groups['time_features']])
            fusion_features.append(time_pred.reshape(-1, 1))
            
        # 原始特征的子集
        important_features = X.select_dtypes(include=[np.number]).iloc[:, :10]  # 取前10个数值特征
        fusion_features.append(important_features.values)
        
        if fusion_features:
            return pd.DataFrame(np.concatenate(fusion_features, axis=1))
        else:
            return pd.DataFrame(X.select_dtypes(include=[np.number]))


class WeightedBlender:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.fitted_models: Dict[str, Any] = {}

    def fit(self, model_dict: Dict[str, Any], X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.fitted_models = model_dict
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for mid, model in self.fitted_models.items():
            w = self.weights.get(mid, 0.0)
            if w == 0:
                continue
            preds.append(w * model.predict(X))
        if not preds:
            raise ValueError("No predictions available in blender. Check weights and models.")
        return np.sum(preds, axis=0)


class StackingBlender:
    """Stacking集成方法，使用元学习器组合多个模型"""
    def __init__(self, meta_model=None):
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.fitted_models: Dict[str, Any] = {}
        
    def fit(self, model_dict: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        self.fitted_models = model_dict
        
        # 生成元特征（各模型的预测结果）
        meta_features = []
        for mid, model in self.fitted_models.items():
            pred = model.predict(X)
            meta_features.append(pred.reshape(-1, 1))
            
        if meta_features:
            meta_X = np.concatenate(meta_features, axis=1)
            self.meta_model.fit(meta_X, y)
            
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = []
        for mid, model in self.fitted_models.items():
            pred = model.predict(X)
            meta_features.append(pred.reshape(-1, 1))
            
        if meta_features:
            meta_X = np.concatenate(meta_features, axis=1)
            return self.meta_model.predict(meta_X)
        else:
            raise ValueError("No fitted models available for prediction")
