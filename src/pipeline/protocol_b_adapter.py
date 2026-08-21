"""Demo 侧 Protocol B 适配器（System A/B 合一 Task 3）。

把 Task 2 的 `RegionPredictionBundle` 经 Task 1 的统一上下文构造器喂进
`build_solver("protocol_b")`，再把 Protocol B 的结果翻译成 demo 流水线统一的
`models / weights / strategy / path_id / yhat`。

**yhat 的来源（Task 3.1 修正的关键点）**

最终 `yhat` **取自引擎实际计算的 `pred_test`**（`ctx.return_predictions=True`），
不再默认由 `df_test[selected] @ weights` 重建。原因是该公式覆盖不了所有可达
分支（行号为 2026-08-20 时的实现）：

1. 交互残差分支被接受时（`weight_meta.interaction_branch.applied=True`，
   protocol_b.py:623）：`pred_test = pred_test + reg_i.predict(X_inter_test)`，
   此时 pred_test **不再**是候选列的线性组合。
2. 其后的 post_adjustment（protocol_b.py:702-716）若被接受，会用
   `df_test[selected] @ w_adj` **整体覆盖** pred_test 并同步更新 weights，
   使其重新变回纯线性组合。

即"线性重建 ≠ 引擎预测"在 **交互应用且 post_adjustment 未应用** 时真实发生
（已由 `tests/test_protocol_b_runtime_predictions.py` 确定性复现）。若沿用重建值
当 yhat，就会出现"trace 和上报 MAE 都正确、但实际输出的是另一条预测"的隐蔽错误。

线性重建作为**核对手段**保留：
- 与引擎上报 MAE 相等 -> `linear_reconstruction_match=True`；
- 不等 -> 记为非线性/后处理分支（`reconcile_note`），但既不报假 MAE、也不用
  重建值覆盖真实预测。

`mae` 字段始终由实际输出的 `yhat` 算出，不照抄引擎数值。引擎未交出预测时
（如被 monkeypatch 的旧式返回）降级为重建值，并以 `yhat_source` 标明来源。

运行时预测不进 `SelectionTrace`、也不留在 `raw` 里（`ProtocolBBackend` 已将其
移出到 `result["predictions"]`），trace 内只保留核对摘要。

本模块不写生产反馈：每次 `select` 使用与 region/scenario 隔离的
`KGFeedbackStore` 实例。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.solver import build_protocol_b_context, build_solver

# 重算 MAE 与 Protocol B 上报 MAE 的一致性容差。
MAE_RECONCILE_TOLERANCE = 1e-8


class DemoProtocolBAdapter:
    """在 demo 数据上调用统一 solver 的 Protocol B 后端。"""

    def __init__(self, tolerance: float = MAE_RECONCILE_TOLERANCE):
        self.tolerance = float(tolerance)

    def _make_feedback_store(self, region: str, horizon: int) -> Any:
        """与 region/horizon 隔离的反馈存储；迁移初期不接生产反馈。"""
        try:
            from src.eval.kg.feedback import KGFeedbackStore
        except Exception:
            return None
        try:
            return KGFeedbackStore()
        except Exception:
            return None

    def select(
        self,
        bundle: Any,
        *,
        region: str,
        horizon: int = 1,
        trace_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """对单个区域跑一次 Protocol B 决策。

        Returns:
            dict，含 `models`、`weights`、`strategy`、`path_id`、`yhat`（引擎实际
            预测）、`yhat_source`、`trace`、`mae`（由 yhat 算出）、`protocol_b_mae`、
            `mae_delta`、`mae_matches_protocol_b`、`linear_reconstruction_mae`、
            `linear_reconstruction_match`、`reconcile_note`、`feedback_store`、`raw`。
        """
        feedback_store = self._make_feedback_store(region, horizon)
        ctx = build_protocol_b_context(
            dataset=region,
            horizon=horizon,
            df_val=bundle.df_val,
            df_test=bundle.df_test,
            df_raw_val=bundle.df_raw_val,
            df_raw_test=bundle.df_raw_test,
            model_cols=list(bundle.model_cols),
            base_model_cols=list(bundle.base_model_cols),
            feedback_store=feedback_store,
            # 显式要求引擎交出真实预测：线性重建覆盖不了所有可达分支。
            return_predictions=True,
        )

        solver = build_solver("protocol_b")
        result, trace = solver.solve(ctx, trace_path=trace_path)

        models: List[str] = list(result.get("models") or [])
        weights: Dict[str, float] = dict(result.get("weights") or {})
        if not models:
            raise ValueError(
                f"DemoProtocolBAdapter: Protocol B selected no model for region {region!r}"
            )

        y_true = np.asarray(bundle.df_test["y"].values, dtype=float)
        protocol_b_mae = self._reported_test_mae(result.get("raw") or {})

        # 权重线性重建：仅作为核对手段保留，**不**用作最终 yhat 的来源。
        linear_yhat = self._weighted_yhat(bundle.df_test, models, weights)
        linear_mae = float(np.mean(np.abs(linear_yhat - y_true)))
        if protocol_b_mae is None:
            linear_match = None
        else:
            linear_match = bool(abs(linear_mae - protocol_b_mae) <= self.tolerance)

        # 优先使用引擎实际计算的 pred_test；线性重建覆盖不了交互残差等分支。
        engine_predictions = result.get("predictions") or {}
        engine_test = engine_predictions.get("test") if isinstance(engine_predictions, dict) else None
        if engine_test is not None and len(engine_test) == len(y_true):
            yhat = np.asarray(engine_test, dtype=float)
            yhat_source = "engine"
        else:
            yhat = linear_yhat
            yhat_source = "linear_reconstruction"

        # 上报 MAE 始终由实际输出的 yhat 算出，绝不照抄引擎数值——否则就会出现
        # "报的 MAE 与实际输出对不上"这一 Task 3.1 要根除的问题。
        mae = float(np.mean(np.abs(yhat - y_true)))
        if protocol_b_mae is None:
            mae_delta = None
            matches = None
        else:
            mae_delta = float(abs(mae - protocol_b_mae))
            matches = bool(mae_delta <= self.tolerance)

        reconcile_note = self._reconcile_note(result.get("raw") or {}, linear_match)

        trace.set_final(models, weights)
        trace.add_stage(
            "DemoProtocolBAdapter",
            inputs={"region": region, "horizon": int(horizon), "n_test": int(len(bundle.df_test))},
            outputs={
                "models": models,
                "weights": weights,
                "strategy": result.get("strategy"),
                "yhat_source": yhat_source,
                "mae": mae,
                "protocol_b_mae": protocol_b_mae,
                "mae_delta": mae_delta,
                "mae_matches_protocol_b": matches,
                "linear_reconstruction_mae": linear_mae,
                "linear_reconstruction_match": linear_match,
                "reconcile_note": reconcile_note,
            },
        )
        if trace_path is not None:
            trace.save_json(trace_path)

        return {
            "models": models,
            "weights": weights,
            "strategy": result.get("strategy"),
            "path_id": result.get("path_id"),
            "yhat": yhat,
            "yhat_source": yhat_source,
            "trace": trace,
            "mae": mae,
            "protocol_b_mae": protocol_b_mae,
            "mae_delta": mae_delta,
            "mae_matches_protocol_b": matches,
            "linear_reconstruction_mae": linear_mae,
            "linear_reconstruction_match": linear_match,
            "reconcile_note": reconcile_note,
            # 返回实例本身供调用方做同一性判断；id() 在对象被回收后会被复用，
            # 不能作为"两次调用用了不同实例"的证据。
            "feedback_store": feedback_store,
            "raw": result.get("raw"),
        }

    @staticmethod
    def _weighted_yhat(df_test, models: List[str], weights: Dict[str, float]) -> np.ndarray:
        missing = [m for m in models if m not in df_test.columns]
        if missing:
            raise ValueError(f"DemoProtocolBAdapter: selected models missing from df_test: {missing}")
        matrix = df_test[models].to_numpy(dtype=float)
        vector = np.array([float(weights.get(m, 0.0)) for m in models], dtype=float)
        return matrix @ vector

    @staticmethod
    def _reported_test_mae(raw: Dict[str, Any]) -> Optional[float]:
        split = raw.get("test") or {}
        if not isinstance(split, dict):
            return None
        value = split.get("mae")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _reconcile_note(raw: Dict[str, Any], linear_match: Optional[bool]) -> Optional[str]:
        """线性重建对不上时说明属于哪类分支；仅作诊断，不改变最终 yhat。"""
        if linear_match is not False:
            return None
        split = raw.get("test") or {}
        weight_meta = split.get("weight_meta") or {} if isinstance(split, dict) else {}
        # 真实键名以 protocol_b.py 的 `ridge_meta["interaction_branch"] = interaction_meta`
        # 为准（weight_meta 即 ridge_meta）；不要凭印象另造键名。
        interaction = weight_meta.get("interaction_branch") or {}
        if isinstance(interaction, dict) and interaction.get("applied"):
            return (
                "interaction_branch_applied: Protocol B 的 pred_test 含交互残差，"
                "不是候选列的线性组合；已改用引擎实际预测，线性重建仅作诊断"
            )
        return "linear_reconstruction_differs_from_protocol_b_reported_mae"


def select_final_output(
    mode: str,
    combinator_result: Any,
    protocol_b_result: Any,
) -> Any:
    """按后端模式决定最终对外输出。

    `protocol_b_shadow` 只影响审计输出，最终结果仍取 combinator——这是影子模式
    的定义，也是"不改变默认输出"这条约束的落点。
    """
    from .main import BACKEND_COMBINATOR, BACKEND_PROTOCOL_B, BACKEND_PROTOCOL_B_SHADOW

    if mode in (BACKEND_COMBINATOR, BACKEND_PROTOCOL_B_SHADOW):
        return combinator_result
    if mode == BACKEND_PROTOCOL_B:
        if protocol_b_result is None:
            raise ValueError("select_final_output: protocol_b mode requires a Protocol B result")
        return protocol_b_result
    raise ValueError(f"select_final_output: unknown backend mode {mode!r}")


def _side_summary(result: Any, elapsed_ms: Optional[float]) -> Dict[str, Any]:
    if result is None:
        return {}
    return {
        "models": list(result.get("models") or []),
        "weights": dict(result.get("weights") or {}),
        "strategy": result.get("strategy"),
        "mae": result.get("mae_recomputed"),
        "elapsed_ms": elapsed_ms,
    }


def build_shadow_comparison(
    *,
    combinator_result: Any,
    protocol_b_result: Any,
    combinator_elapsed_ms: Optional[float] = None,
    protocol_b_elapsed_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """影子模式的审计摘要：两套候选、权重、耗时、MAE 及差异。

    只用于写 trace / 报告，不参与最终输出选择。
    """
    combinator_side = _side_summary(combinator_result, combinator_elapsed_ms)
    protocol_b_side = _side_summary(protocol_b_result, protocol_b_elapsed_ms)

    mae_a = combinator_side.get("mae")
    mae_b = protocol_b_side.get("mae")
    mae_delta = None
    if mae_a is not None and mae_b is not None:
        mae_delta = float(mae_b) - float(mae_a)

    return {
        "combinator": combinator_side,
        "protocol_b": protocol_b_side,
        "diff": {
            "selection_changed": combinator_side.get("models") != protocol_b_side.get("models"),
            "weights_changed": combinator_side.get("weights") != protocol_b_side.get("weights"),
            "mae_delta": mae_delta,
        },
        # 影子模式下最终输出恒为 combinator；显式写出来便于审计时一眼确认。
        "final_output_from": "combinator",
    }
