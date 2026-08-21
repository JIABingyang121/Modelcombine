"""Uncertainty utilities for auditable model scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.core.solver.context import SolveContext
from src.core.trace import SelectionTrace


@dataclass
class PredictionInterval:
    model_id: str
    yhat: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    method: str
    alpha: float
    score: float

    @property
    def width(self) -> np.ndarray:
        return np.maximum(self.upper - self.lower, 0.0)


class UncertaintyEstimator:
    """Estimate predictive intervals and normalized uncertainty scores."""

    def __init__(
        self,
        residuals_by_model: Optional[Mapping[str, Sequence[float]]] = None,
        min_scale: float = 1e-8,
    ):
        self.residuals_by_model = dict(residuals_by_model or {})
        self.min_scale = float(min_scale)

    def estimate(
        self,
        model: Any,
        X: Any,
        *,
        model_id: str,
        residuals: Optional[Sequence[float]] = None,
        alpha: float = 0.1,
    ) -> PredictionInterval:
        self._validate_alpha(alpha)
        if hasattr(model, "predict_interval"):
            yhat, lower, upper = model.predict_interval(X, alpha=alpha)
            method = "native"
        else:
            yhat = model.predict(X)
            resolved_residuals = residuals
            if resolved_residuals is None:
                resolved_residuals = self.residuals_by_model.get(model_id)
            lower, upper, method = self._fallback_interval(
                yhat,
                resolved_residuals,
                alpha=alpha,
            )

        yhat_arr = self._as_1d(yhat)
        lower_arr = self._as_1d(lower)
        upper_arr = self._as_1d(upper)
        lower_arr, upper_arr = self._ordered_bounds(lower_arr, upper_arr)
        return PredictionInterval(
            model_id=model_id,
            yhat=yhat_arr,
            lower=lower_arr,
            upper=upper_arr,
            method=method,
            alpha=float(alpha),
            score=self._score(yhat_arr, lower_arr, upper_arr),
        )

    def estimate_many(
        self,
        models: Mapping[str, Any],
        X: Any,
        *,
        residuals_by_model: Optional[Mapping[str, Sequence[float]]] = None,
        alpha: float = 0.1,
    ) -> Dict[str, PredictionInterval]:
        residuals_by_model = residuals_by_model or {}
        return {
            model_id: self.estimate(
                model,
                X,
                model_id=model_id,
                residuals=residuals_by_model.get(model_id),
                alpha=alpha,
            )
            for model_id, model in models.items()
        }

    def _fallback_interval(
        self,
        yhat: Sequence[float],
        residuals: Optional[Sequence[float]],
        *,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        yhat_arr = self._as_1d(yhat)
        residual_arr = self._clean_residuals(residuals)
        if residual_arr.size == 0:
            return yhat_arr.copy(), yhat_arr.copy(), "point_estimate"

        lo, hi = np.quantile(residual_arr, [alpha / 2.0, 1.0 - alpha / 2.0])
        return yhat_arr + lo, yhat_arr + hi, "residual_bootstrap"

    def _score(self, yhat: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        width = np.maximum(upper - lower, 0.0)
        finite_width = width[np.isfinite(width)]
        finite_yhat = yhat[np.isfinite(yhat)]
        scale = self.min_scale
        if finite_yhat.size:
            scale = max(
                float(np.mean(np.abs(finite_yhat))),
                float(np.std(finite_yhat)),
                self.min_scale,
            )
        return float(np.mean(finite_width) / scale) if finite_width.size else 0.0

    @staticmethod
    def _as_1d(values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1)

    @staticmethod
    def _clean_residuals(values: Optional[Sequence[float]]) -> np.ndarray:
        if values is None:
            return np.asarray([], dtype=float)
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _ordered_bounds(
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.minimum(lower, upper), np.maximum(lower, upper)

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be between 0 and 1")


class UncertaintyGate:
    """Solver post-stage that can apply an auditable uncertainty bypass."""

    def __init__(
        self,
        threshold: float,
        *,
        min_keep: int = 1,
        scores: Optional[Mapping[str, float]] = None,
        estimator: Optional[UncertaintyEstimator] = None,
        alpha: float = 0.1,
    ):
        self.threshold = float(threshold)
        self.min_keep = int(min_keep)
        if self.min_keep < 0:
            raise ValueError("min_keep must be non-negative")
        self.scores = dict(scores or {})
        self.estimator = estimator or UncertaintyEstimator()
        self.alpha = float(alpha)

    def apply(self, ctx: SolveContext, trace: SelectionTrace) -> None:
        result = dict(ctx.extras.get("result") or {})
        selected = list(result.get("models") or ctx.model_cols)
        scores = self._resolve_scores(ctx, selected)
        ctx.uncertainty.update(scores)

        keep, removed = self._split_by_threshold(selected, scores)
        bypass_applied = bool(removed)
        if bypass_applied:
            result["models"] = keep
            result["weights"] = self._renormalize_weights(
                keep,
                dict(result.get("weights") or {}),
            )
            ctx.extras["result"] = result
            for model_id in removed:
                trace.reject(
                    model_id,
                    "uncertainty_bypass: "
                    f"score={scores.get(model_id, 0.0):.6g} exceeds "
                    f"threshold={self.threshold:.6g}; decision changed outside graph inference",
                )

        trace.add_stage(
            "UncertaintyEstimate",
            inputs={
                "threshold": self.threshold,
                "min_keep": self.min_keep,
                "models": selected,
            },
            outputs={
                "uncertainty": {mid: scores.get(mid, 0.0) for mid in selected},
                "kept": keep,
                "removed": removed,
            },
            metadata={
                "bypass_applied": bypass_applied,
                "decision_authority": (
                    "uncertainty_bypass" if bypass_applied else "audit_only"
                ),
            },
        )

    def _resolve_scores(
        self,
        ctx: SolveContext,
        selected: Sequence[str],
    ) -> Dict[str, float]:
        result = ctx.extras.get("result") or {}
        scores: Dict[str, float] = {
            model_id: float(ctx.uncertainty.get(model_id, 0.0))
            for model_id in selected
        }
        for source in (
            result.get("uncertainty", {}),
            ctx.extras.get("uncertainty", {}),
            self.scores,
        ):
            for model_id, score in dict(source or {}).items():
                if model_id in selected:
                    scores[model_id] = float(score)

        models = ctx.extras.get("uncertainty_models")
        X = ctx.extras.get("uncertainty_X")
        if models is not None and X is not None:
            residuals_by_model = ctx.extras.get("uncertainty_residuals", {})
            intervals = self.estimator.estimate_many(
                {mid: models[mid] for mid in selected if mid in models},
                X,
                residuals_by_model=residuals_by_model,
                alpha=self.alpha,
            )
            ctx.extras["uncertainty_intervals"] = intervals
            for model_id, interval in intervals.items():
                scores[model_id] = interval.score
        return scores

    def _split_by_threshold(
        self,
        selected: Sequence[str],
        scores: Mapping[str, float],
    ) -> tuple[list[str], list[str]]:
        keep = [mid for mid in selected if scores.get(mid, 0.0) <= self.threshold]
        if len(keep) < min(self.min_keep, len(selected)):
            keep_set = set(keep)
            needed = min(self.min_keep, len(selected)) - len(keep)
            fallback = sorted(
                (mid for mid in selected if mid not in keep_set),
                key=lambda mid: (scores.get(mid, 0.0), selected.index(mid)),
            )
            keep.extend(fallback[:needed])

        keep_set = set(keep)
        removed = [mid for mid in selected if mid not in keep_set]
        return keep, removed

    @staticmethod
    def _renormalize_weights(
        keep: Sequence[str],
        weights: Mapping[str, float],
    ) -> Dict[str, float]:
        if not keep:
            return {}
        kept_weights = {mid: float(weights.get(mid, 0.0)) for mid in keep}
        total = sum(kept_weights.values())
        if total <= 0.0:
            equal = 1.0 / len(keep)
            return {mid: equal for mid in keep}
        return {mid: value / total for mid, value in kept_weights.items()}
