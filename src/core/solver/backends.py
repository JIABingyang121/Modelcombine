"""Combination backend adapters for the logical solver."""
from __future__ import annotations

from typing import Any, Dict

from src.eval.kg.config import RUNTIME_PREDICTIONS_KEY

from ..trace import SelectionTrace
from .context import CombinationBackend, SolveContext


class ProtocolBBackend(CombinationBackend):
    """Delegate to src.eval.kg.protocol_b.kg_combination_with_features.

    Protocol B is a full combination engine over prediction DataFrames, so this backend
    normalizes its result shape rather than trying to expose its internal guard chain as stages.
    """

    name = "protocol_b"

    def combine(self, ctx: SolveContext, trace: SelectionTrace) -> Dict[str, Any]:
        missing = []
        if ctx.df_val is None:
            missing.append("df_val")
        if ctx.df_test is None:
            missing.append("df_test")
        if not ctx.model_cols:
            missing.append("model_cols")
        if missing:
            raise ValueError(f"ProtocolBBackend requires: {', '.join(missing)}")

        from src.eval.kg.protocol_b import kg_combination_with_features

        raw = kg_combination_with_features(
            ctx.df_val,
            ctx.df_test,
            ctx.df_raw_val,
            ctx.df_raw_test,
            ctx.model_cols,
            ctx.horizon,
            dataset_name=ctx.dataset_name,
            base_model_cols=ctx.base_model_cols,
            feedback_store=ctx.feedback_store,
            return_predictions=bool(getattr(ctx, "return_predictions", False)),
        )
        normalized = self._normalize(raw)
        audit = self._guard_audit(raw)
        trace.consider(list(ctx.model_cols))
        for model_id, reason in audit["removed_models"].items():
            trace.reject(model_id, reason)
        trace.add_stage(
            "ProtocolBBackend",
            inputs={
                "n_models": len(ctx.model_cols),
                "horizon": ctx.horizon,
                "dataset_name": ctx.dataset_name,
                "candidates": list(ctx.model_cols),
            },
            outputs={
                "protocol": normalized.get("protocol"),
                "models": normalized.get("models", []),
                "strategy": normalized.get("strategy"),
                # 不再只记录最终模型列表：保留 B 的原始候选、guard 移除原因和回退目标，
                # 以便在被回退时仍能回答"B 本来想选什么、为什么没用它"。
                "protocol_b_candidates": audit["protocol_b_candidates"],
                "protocol_b_candidate_weights": audit["protocol_b_candidate_weights"],
                "fallback_target": audit["fallback_target"],
                "fallback_reason": audit["fallback_reason"],
                "removed_models": audit["removed_models"],
            },
        )
        return normalized

    @staticmethod
    def _guard_audit(raw: Dict[str, Any]) -> Dict[str, Any]:
        """从 Protocol B 返回值中提取 guard 审计信息。

        字段位置以 src/eval/kg/protocol_b.py 的真实产出为准；各分支产出的字段
        并不齐全（例如回退到 best_single 时不带 protocol_b_selection_meta），
        因此全部按"存在才取"处理，缺失记为 None/空而不是伪造。
        """
        split = raw.get("test") or raw.get("val") or {}
        if not isinstance(split, dict):
            split = {}
        weight_meta = split.get("weight_meta") or {}
        if not isinstance(weight_meta, dict):
            weight_meta = {}

        guard = weight_meta.get("protocol_b_guard") or {}
        guard_config = weight_meta.get("guard_config") or {}
        fallback_target = None
        fallback_reason = None
        if isinstance(guard, dict) and guard.get("fallback_target") is not None:
            fallback_target = guard.get("fallback_target")
            fallback_reason = guard.get("reason")
        elif isinstance(guard_config, dict):
            fallback_target = guard_config.get("final_fallback_target")
            fallback_reason = guard_config.get("final_fallback_reason")

        removed: Dict[str, str] = {}
        selection_meta = weight_meta.get("protocol_b_selection_meta") or {}
        if isinstance(selection_meta, dict):
            stability = selection_meta.get("stability") or {}
            if isinstance(stability, dict):
                for model_id in stability.get("removed_models") or []:
                    removed[str(model_id)] = "protocol_b_stability_filter"

        return {
            "protocol_b_candidates": list(split.get("selected_models_b_candidate") or []),
            "protocol_b_candidate_weights": dict(split.get("weights_b_candidate") or {}),
            "fallback_target": fallback_target,
            "fallback_reason": fallback_reason,
            "removed_models": removed,
        }

    def _normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        split = raw.get("test") or raw.get("val") or {}
        models = list(split.get("selected_models", []))
        weights = dict(split.get("weights", {}))
        protocol = raw.get("protocol", "protocol_b")
        # 运行时预测（若引擎按 ctx.return_predictions 交出）从 raw 中**移出**，
        # 保证 raw 始终是可 JSON 序列化的、不含数组的实验载荷；预测另走
        # normalized["predictions"]，由调用方直接取用，绝不进 trace/实验 JSON。
        predictions = raw.pop(RUNTIME_PREDICTIONS_KEY, None)
        return {
            "models": models,
            "weights": weights,
            "strategy": protocol,
            "path_id": protocol,
            "protocol": protocol,
            "predictions": predictions,
            "raw": raw,
        }


class CombinatorBackend(CombinationBackend):
    """Adapter around the legacy PowerModelCombinator.

    This is a golden-reference bridge: it delegates to select_optimal_path and normalizes the
    result without changing the old combinator behavior.
    """

    name = "combinator"

    def __init__(self, combinator: Any = None):
        self.combinator = combinator

    def combine(self, ctx: SolveContext, trace: SelectionTrace) -> Dict[str, Any]:
        if not ctx.model_cols:
            raise ValueError("CombinatorBackend requires model_cols")

        combinator = self._get_combinator()
        if ctx.historical_scenarios and hasattr(combinator, "set_historical_scenarios"):
            combinator.set_historical_scenarios(ctx.historical_scenarios)

        similar_scenarios = self._similar_as_tuples(ctx)
        raw = combinator.select_optimal_path(
            scenario_signature=dict(ctx.scenario.signature),
            available_models=list(ctx.model_cols),
            constraints=dict(ctx.constraints),
            model_graph=ctx.model_graph,
            similar_scenarios=similar_scenarios,
            actual_data_columns=set(ctx.available_features),
        )
        normalized = self._normalize(raw)
        reasoning_paths = self._reasoning_paths(ctx, normalized, trace)
        outputs = {
            "models": normalized.get("models", []),
            "strategy": normalized.get("strategy"),
            "path_id": normalized.get("path_id"),
        }
        if reasoning_paths:
            outputs["reasoning_paths"] = reasoning_paths
        trace.add_stage(
            "CombinatorBackend",
            inputs={
                "n_models": len(ctx.model_cols),
                "constraints": dict(ctx.constraints),
                "n_similar": len(similar_scenarios or []),
            },
            outputs=outputs,
        )
        return normalized

    def _reasoning_paths(
        self,
        ctx: SolveContext,
        normalized: Dict[str, Any],
        trace: SelectionTrace,
    ) -> list[Dict[str, Any]]:
        """可解释推理链：为选中/候选 Path 计算多跳评分并写入 trace（尽力而为，不阻断调度）。"""
        if ctx.model_graph is None:
            return []
        try:
            from src.graph.path_reasoning import top_reasoning_paths

            scenario_id = getattr(ctx.scenario, "scenario_id", "") or ""
            results = top_reasoning_paths(
                ctx.model_graph,
                scenario_id,
                available_features=set(ctx.available_features),
                include_path_ids=[normalized.get("path_id")],
            )
        except Exception:
            return []
        for item in results:
            for ref in item.evidence_refs:
                trace.add_evidence_ref(ref)
        return [item.to_dict() for item in results]

    def _get_combinator(self) -> Any:
        if self.combinator is None:
            from src.selector.combinator import PowerModelCombinator

            self.combinator = PowerModelCombinator()
        return self.combinator

    def _similar_as_tuples(self, ctx: SolveContext) -> list[tuple[str, float]]:
        if ctx.similar_scenarios:
            return list(ctx.similar_scenarios)
        converted = []
        for record in ctx.similar:
            scenario_id = record.get("scenario_id")
            score = record.get("_score", record.get("score", 0.0))
            if scenario_id is not None:
                converted.append((scenario_id, float(score)))
        return converted

    def _normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "models": list(raw.get("models", [])),
            "weights": dict(raw.get("weights", {})),
            "strategy": raw.get("strategy", ""),
            "path_id": raw.get("path_id", ""),
            "raw": raw,
        }
