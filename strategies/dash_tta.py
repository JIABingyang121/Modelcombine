from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .rl_qms import BaseStrategy, StrategyOutput


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -10.0, 10.0)))


def dash_tta_v2(
    P: np.ndarray,
    y: np.ndarray,
    active_idx: List[int],
    P_baseline: np.ndarray,
    delay: int = 1,
    scale: float = 1.0,
    eta_base: float = 0.5,
    eta_max: float = 3.0,
    alpha_base: float = 0.01,
    alpha_max: float = 0.20,
    beta_hi: float = 0.9,
    beta_lo: float = 0.5,
    clip: float = 0.85,
    drift_score: Optional[np.ndarray] = None,
    drift_tau: float = 0.5,
    drift_kappa: float = 10.0,
    window: int = 48,
    fallback_delta: float = 0.05,
    # Fix1: 内部误差漂移信号参数
    internal_drift_alpha_fast: float = 0.15,
    internal_drift_alpha_slow: float = 0.02,
    internal_drift_tau: float = 0.10,
    internal_drift_kappa: float = 8.0,
    # Fix2: Soft Fallback 参数
    fallback_soft_strength: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """DASH-TTA v2: Drift-aware safe hedge with delayed online adaptation."""
    if P_baseline is None:
        raise ValueError("P_baseline is required for worst-case safety fallback.")

    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    P_baseline = np.asarray(P_baseline, dtype=float).reshape(-1)
    if P.ndim == 1:
        P = P[:, None]
    if len(P) != len(y) or len(P) != len(P_baseline):
        raise ValueError(
            f"length mismatch: len(P)={len(P)}, len(y)={len(y)}, len(P_baseline)={len(P_baseline)}"
        )

    A = list(active_idx)
    if not A:
        raise ValueError("active_idx must not be empty.")
    if min(A) < 0 or max(A) >= P.shape[1]:
        raise ValueError("active_idx out of bound for P columns.")

    M = len(A)
    M_total = M + 1  # +1 structural baseline expert
    w = np.ones(M_total, dtype=float) / float(M_total)

    T = len(y)
    yhat = np.zeros(T, dtype=np.float32)
    weights_log = np.zeros((T, M_total), dtype=np.float32)
    loss_buf: List[Optional[np.ndarray]] = [None] * T
    ens_loss_buf = np.zeros(T, dtype=float)
    base_loss_buf = np.zeros(T, dtype=float)
    E_t = np.zeros(M_total, dtype=float)

    recent_loss_ens = deque(maxlen=max(8, int(window)))
    recent_loss_base = deque(maxlen=max(8, int(window)))
    fallback_count = 0
    gate_history: List[float] = []

    safe_delay = max(1, int(delay))
    safe_scale = float(scale) if np.isfinite(scale) and float(scale) > 1e-12 else 1.0

    if drift_score is not None:
        drift_score = np.asarray(drift_score, dtype=float).reshape(-1)
        if len(drift_score) != T:
            drift_score = None

    # Fix1: 内部误差漂移信号 EMA（当 drift_score 缺失或死区时启用）
    # 用快慢双 EMA 的比值近似局部误差突增程度作为漂移代理
    # 死区判断：列存在但全为常数（std < 1e-4）时，gate ≈ sigmoid(-5) ≈ 0.007，等同失效
    _ema_fast = float("nan")
    _ema_slow = float("nan")
    _use_internal_drift = (drift_score is None) or (
        float(np.nanstd(drift_score)) < 1e-4
    )

    for t in range(T):
        if t >= safe_delay and loss_buf[t - safe_delay] is not None:
            fb_idx = t - safe_delay
            loss_fb = loss_buf[fb_idx]

            if _use_internal_drift:
                # 用延迟反馈时刻的集成损失更新内部漂移 EMA
                fb_ens_loss = float(ens_loss_buf[fb_idx])
                if not np.isfinite(_ema_fast):
                    _ema_fast = fb_ens_loss
                    _ema_slow = fb_ens_loss
                else:
                    _ema_fast = internal_drift_alpha_fast * fb_ens_loss + (1.0 - internal_drift_alpha_fast) * _ema_fast
                    _ema_slow = internal_drift_alpha_slow * fb_ens_loss + (1.0 - internal_drift_alpha_slow) * _ema_slow
                # 漂移信号：快 EMA 与慢 EMA 的比值 - 1（误差相对基线突增比例）
                internal_d = max(0.0, _ema_fast / (_ema_slow + 1e-12) - 1.0)
                d_score = internal_d
                gate = float(sigmoid(internal_drift_kappa * (d_score - internal_drift_tau)))
            else:
                d_score = float(drift_score[fb_idx])
                gate = float(sigmoid(drift_kappa * (d_score - drift_tau)))
            gate_history.append(gate)

            a_t = alpha_base + (alpha_max - alpha_base) * gate
            e_t = eta_base + (eta_max - eta_base) * gate
            beta_t = beta_hi - (beta_hi - beta_lo) * gate

            E_t = beta_t * E_t + (1.0 - beta_t) * loss_fb
            w = w * np.exp(-e_t * E_t)
            w = w / (w.sum() + 1e-12)

            # Dynamic fixed-share
            w = (1.0 - a_t) * w + a_t * (np.ones(M_total, dtype=float) / float(M_total))

            # Weight clipping
            w = np.minimum(w, clip)
            w = w / (w.sum() + 1e-12)

            # Strictly delayed fallback monitor: only consume feedback that has arrived.
            recent_loss_ens.append(float(ens_loss_buf[fb_idx]))
            recent_loss_base.append(float(base_loss_buf[fb_idx]))
            if len(recent_loss_ens) == recent_loss_ens.maxlen:
                ens_mean = float(np.mean(recent_loss_ens))
                base_mean = float(np.mean(recent_loss_base))
                if ens_mean > base_mean * (1.0 + fallback_delta):
                    fallback_count += 1
                    # Fix2: Soft Fallback——按 fallback_soft_strength 插值向 baseline 靠拢，
                    # 而非硬重置为 [0,...,0,1]，避免长 horizon 下持续振荡
                    w_baseline = np.zeros(M_total, dtype=float)
                    w_baseline[-1] = 1.0
                    w = (1.0 - fallback_soft_strength) * w + fallback_soft_strength * w_baseline
                    w = w / (w.sum() + 1e-12)
                    E_t.fill(0.0)
                    recent_loss_ens.clear()
                    recent_loss_base.clear()

        p_all_t = np.append(P[t, A], P_baseline[t])
        weights_log[t] = w.astype(np.float32)
        yhat[t] = float(np.dot(w, p_all_t))

        loss_t = np.abs(y[t] - p_all_t) / (safe_scale + 1e-12)
        loss_buf[t] = loss_t.astype(float)

        ens_loss_val = float(np.abs(y[t] - yhat[t]) / (safe_scale + 1e-12))
        base_loss_val = float(loss_t[-1])
        ens_loss_buf[t] = ens_loss_val
        base_loss_buf[t] = base_loss_val

    gate_arr = np.array(gate_history, dtype=float) if gate_history else np.zeros(1, dtype=float)
    meta = {
        "deployable": True,
        "update_delay": safe_delay,
        "update_after_observe": True,
        "drift_adapt": True,
        "fallback_count": int(fallback_count),
        "gate_mean": float(np.mean(gate_arr)),
        "gate_p95": float(np.percentile(gate_arr, 95)),
        "drift_signal_source": "internal_ema" if _use_internal_drift else "external_col",
    }
    return yhat, weights_log, meta


class DASHTTAStrategy(BaseStrategy):
    """DASH-TTA strategy wrapper following the project BaseStrategy interface."""

    name = "dash_tta"

    def __init__(
        self,
        *,
        delay: int = 1,
        scale: float = 1.0,
        eta_base: float = 0.5,
        eta_max: float = 3.0,
        alpha_base: float = 0.01,
        alpha_max: float = 0.20,
        beta_hi: float = 0.9,
        beta_lo: float = 0.5,
        clip: float = 0.85,
        drift_tau: float = 0.5,
        drift_kappa: float = 10.0,
        window: int = 48,
        fallback_delta: float = 0.05,
        active_models: Optional[List[str]] = None,
        drift_score_col: str = "drift_score",
        baseline_mode: str = "simple_avg",
        # Fix3: horizon-adaptive 控制（默认 True，自动按 delay 调整 window/fallback_delta）
        horizon_adaptive: bool = True,
        # Fix2: Soft Fallback 强度（1=硬重置到 baseline，0=权重不变，推荐 0.7）
        # 公式：w = (1-s)*w + s*w_baseline，s=1 时等价于硬重置
        fallback_soft_strength: float = 0.7,
    ):
        self.delay = max(1, int(delay))
        self.scale = float(scale)
        self.eta_base = float(eta_base)
        self.eta_max = float(eta_max)
        self.alpha_base = float(alpha_base)
        self.alpha_max = float(alpha_max)
        self.beta_hi = float(beta_hi)
        self.beta_lo = float(beta_lo)
        self.clip = float(clip)
        self.drift_tau = float(drift_tau)
        self.drift_kappa = float(drift_kappa)
        self.fallback_delta = float(fallback_delta)
        self.fallback_soft_strength = float(np.clip(fallback_soft_strength, 0.0, 1.0))
        self.active_models = active_models
        self.drift_score_col = str(drift_score_col)
        self.baseline_mode = str(baseline_mode).strip().lower()
        if self.baseline_mode not in {"simple_avg", "best_model"}:
            self.baseline_mode = "simple_avg"
        # Fix3: Horizon-adaptive window 和 fallback_delta
        # window: H1→max(48,6), H6→max(48,36), H24→max(48,144)=144
        # fallback_delta: H24 放宽到 0.125，避免高方差环境误触发
        if horizon_adaptive:
            self.window = max(max(8, int(window)), self.delay * 6)
            if self.delay >= 24:
                self.fallback_delta = max(self.fallback_delta, 0.125)
            elif self.delay >= 6:
                self.fallback_delta = max(self.fallback_delta, 0.075)
        else:
            self.window = max(8, int(window))

        self._active_indices: Optional[List[int]] = None
        self._active_to_full: Dict[int, int] = {}
        self._best_active_idx: Optional[int] = None
        self._last_train_mae: Optional[float] = None

    def fit(self, P_val, y_val, ctx_val=None, model_names=None):
        P_val = np.asarray(P_val, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        if P_val.ndim == 1:
            P_val = P_val[:, None]
        if P_val.size == 0 or len(P_val) != len(y_val):
            return self

        M = P_val.shape[1]
        if self.active_models is not None and model_names is not None:
            active = [i for i, m in enumerate(model_names) if m in self.active_models]
            self._active_indices = active if active else list(range(M))
        else:
            self._active_indices = list(range(M))
        self._active_to_full = {i: idx for i, idx in enumerate(self._active_indices)}

        P_active = P_val[:, self._active_indices]
        mae = np.nanmean(np.abs(P_active - y_val[:, None]), axis=0)
        if np.isfinite(mae).any():
            self._best_active_idx = int(np.nanargmin(mae))
            self._last_train_mae = float(np.nanmin(mae))
        else:
            self._best_active_idx = 0
            self._last_train_mae = None

        if not np.isfinite(self.scale) or self.scale <= 1e-12:
            val_scale = float(np.nanmean(np.abs(y_val - np.nanmean(P_active, axis=1))))
            self.scale = val_scale if np.isfinite(val_scale) and val_scale > 1e-12 else 1.0
        return self

    def _resolve_active_indices(self, M: int, model_names=None) -> List[int]:
        if self._active_indices is not None:
            return self._active_indices
        if self.active_models is not None and model_names is not None:
            active = [i for i, m in enumerate(model_names) if m in self.active_models]
            if active:
                self._active_indices = active
                self._active_to_full = {i: idx for i, idx in enumerate(active)}
                return active
        self._active_indices = list(range(M))
        self._active_to_full = {i: i for i in range(M)}
        return self._active_indices

    def _extract_drift_score(self, ctx_test) -> Optional[np.ndarray]:
        if ctx_test is None:
            return None
        df = ctx_test if isinstance(ctx_test, pd.DataFrame) else pd.DataFrame(ctx_test)
        if self.drift_score_col not in df.columns:
            return None
        col = pd.to_numeric(df[self.drift_score_col], errors="coerce")
        if col.isna().all():
            return None
        return col.fillna(col.median()).values.astype(float)

    def _build_baseline(
        self,
        P_test: np.ndarray,
        active_idx: List[int],
        baseline_series: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, str]:
        P_active = P_test[:, active_idx]
        fallback = np.nanmean(P_active, axis=1).astype(float)
        if baseline_series is not None:
            ext = np.asarray(baseline_series, dtype=float).reshape(-1)
            if len(ext) != len(P_test):
                raise ValueError(
                    f"baseline_series length mismatch: {len(ext)} vs {len(P_test)}"
                )
            out = ext.copy()
            mask = ~np.isfinite(out)
            if mask.any():
                out[mask] = fallback[mask]
            return out.astype(float), "external_series"
        if self.baseline_mode == "best_model" and self._best_active_idx is not None:
            bid = min(max(int(self._best_active_idx), 0), P_active.shape[1] - 1)
            return P_active[:, bid].astype(float), "best_model"
        return fallback, "simple_avg"

    def predict(
        self,
        P_test,
        y_test=None,
        ctx_test=None,
        model_names=None,
        baseline_series=None,
    ) -> StrategyOutput:
        if y_test is None:
            raise ValueError("DASH-TTA requires delayed true feedback y_test for online adaptation.")

        P_test = np.asarray(P_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        if P_test.ndim == 1:
            P_test = P_test[:, None]
        if len(P_test) != len(y_test):
            raise ValueError(f"P_test/y_test length mismatch: {len(P_test)} vs {len(y_test)}")

        T, M_full = P_test.shape
        if T == 0:
            return StrategyOutput(
                y_pred_test=np.zeros((0,), dtype=np.float32),
                weights_log=np.zeros((0, M_full), dtype=np.float32),
                meta={"deployable": True, "empty_input": True},
                chosen_models=np.zeros((0,), dtype=np.int32),
            )

        active_idx = self._resolve_active_indices(M_full, model_names=model_names)
        P_baseline, baseline_source = self._build_baseline(
            P_test,
            active_idx,
            baseline_series=baseline_series,
        )
        drift_score = self._extract_drift_score(ctx_test)

        yhat, weights_log_ext, algo_meta = dash_tta_v2(
            P=P_test,
            y=y_test,
            active_idx=active_idx,
            P_baseline=P_baseline,
            delay=self.delay,
            scale=self.scale,
            eta_base=self.eta_base,
            eta_max=self.eta_max,
            alpha_base=self.alpha_base,
            alpha_max=self.alpha_max,
            beta_hi=self.beta_hi,
            beta_lo=self.beta_lo,
            clip=self.clip,
            drift_score=drift_score,
            drift_tau=self.drift_tau,
            drift_kappa=self.drift_kappa,
            window=self.window,
            fallback_delta=self.fallback_delta,
            fallback_soft_strength=self.fallback_soft_strength,
        )

        M_active = len(active_idx)
        weights_active = weights_log_ext[:, :M_active]
        baseline_weights = weights_log_ext[:, M_active]
        weights_full = np.zeros((T, M_full), dtype=np.float32)
        for i_active, i_full in enumerate(active_idx):
            weights_full[:, i_full] = weights_active[:, i_active]

        chosen_full = np.argmax(weights_full, axis=1).astype(np.int32)
        baseline_dominant = baseline_weights > np.max(weights_active, axis=1)
        chosen_full[baseline_dominant] = -1

        meta = {
            "strategy": self.name,
            "active_models_count": int(M_active),
            "total_models_count": int(M_full),
            "baseline_mode": self.baseline_mode,
            "baseline_source": baseline_source,
            "baseline_dominant_ratio": float(np.mean(baseline_dominant)),
            "baseline_weight_mean": float(np.mean(baseline_weights)),
            "train_mae_best_active": self._last_train_mae,
            "drift_score_col": self.drift_score_col,
            "cost_assumption": "online_delayed_feedback",
        }
        meta.update(algo_meta)
        return StrategyOutput(yhat.astype(np.float32), weights_full, meta, chosen_full)
