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
            # §11#7 生产接线：把当前场景的关系强度交给引擎。
            # 此前未传，引擎内部新建空图，关系项在真实路径上恒为中性。
            relation_graph=ctx.model_graph,
            relation_scenario_id=getattr(ctx.scenario, "scenario_id", None),
        )
        normalized = self._normalize(raw)
        audit = self._guard_audit(raw)
        relation_audit = self._relation_scoring_audit(raw)
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
                # 关系强度如何参与本次评分与排序（§11#7）
                "relation_strength": relation_audit["relation_strength"],
                "candidate_ranking": relation_audit["candidate_ranking"],
                "candidate_scores": relation_audit["candidate_scores"],
                # 逐步选择轨迹：每步选了谁、CV MAE 多少、是否发生并列。
                # 缺了它，两次运行选出不同组合时无法定位到分歧发生在哪一步。
                "stepwise": relation_audit["stepwise"],
                # stepwise 之后 reasoning/hybrid 是否覆盖了候选集合。
                "selection_flow": relation_audit["selection_flow"],
            },
        )
        return normalized

    @staticmethod
    def _relation_scoring_audit(raw: Dict[str, Any]) -> Dict[str, Any]:
        """提取关系强度评分项、候选排序与最终选择（§11#7 可审计性要求）。

        关系强度写进图谱后必须能回溯到决策，否则又会回到"信号已更新、决策不消费"
        的状态。回退分支不带 selection meta 时记为 None，不抛错。
        """
        # 真实引擎把 model_scores_b 与 selection meta 写在 **val** 里
        # （protocol_b.py 的最终返回与 guarded_val 均如此），test 只带指标与选择。
        # 此前一律先读 test，导致生产路径上候选得分/排序恒为空。
        def _pick(key: str):
            for split_name in ("val", "test"):
                split = raw.get(split_name)
                if isinstance(split, dict) and split.get(key):
                    return split[key]
            return None

        split = raw.get("test") or raw.get("val") or {}
        if not isinstance(split, dict):
            split = {}
        weight_meta = _pick("weight_meta") or {}
        if not isinstance(weight_meta, dict):
            weight_meta = {}
        selection_meta = weight_meta.get("protocol_b_selection_meta") or {}
        if not isinstance(selection_meta, dict):
            selection_meta = {}

        relation = selection_meta.get("relation_strength")
        scores = _pick("model_scores_b") or {}
        # 得分相同时按名称定序：否则审计出来的排序取决于字典插入顺序，
        # 同一份结果两次读可能给出不同排序，反而妨碍定位。
        ranking = (
            [m for m, _ in sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))]
            if isinstance(scores, dict) else []
        )
        stepwise = selection_meta.get("stepwise_meta")
        if isinstance(stepwise, dict):
            stepwise = {
                **stepwise,
                "alpha": selection_meta.get("stepwise_alpha"),
                "min_improve_ratio": selection_meta.get("stepwise_min_improve_ratio"),
            }
        return {
            "relation_strength": relation if isinstance(relation, dict) else None,
            "candidate_scores": dict(scores) if isinstance(scores, dict) else {},
            "candidate_ranking": ranking,
            "stepwise": stepwise if isinstance(stepwise, dict) else None,
            "score_components": (
                selection_meta.get("score_components")
                if isinstance(selection_meta.get("score_components"), dict)
                else None
            ),
            "pair_diagnostics": (
                selection_meta.get("pair_diagnostics")
                if isinstance(selection_meta.get("pair_diagnostics"), dict)
                else None
            ),
            "selection_flow": (
                selection_meta.get("selection_flow")
                if isinstance(selection_meta.get("selection_flow"), dict)
                else None
            ),
            "final_selection": list(split.get("selected_models") or []),
        }

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

    @staticmethod
    def _relation_feedback_audit(raw: Dict[str, Any]) -> Dict[str, Any]:
        """提取带符号关系反馈证据（Task 8.3 Task 5）。

        引擎在 raw 顶层与 val.weight_meta 均写入 relation_feedback；缺失时返回
        不可用 payload 而非伪造 eligible=True。
        """
        fb = raw.get("relation_feedback")
        if isinstance(fb, dict):
            return fb
        for split_name in ("val", "test"):
            split = raw.get(split_name)
            if isinstance(split, dict):
                wm = split.get("weight_meta")
                if isinstance(wm, dict) and isinstance(wm.get("relation_feedback"), dict):
                    return wm["relation_feedback"]
        return {
            "eligible": False,
            "skip_reason": "no_relation_feedback",
            "deadband": 0.005,
            "evidence_mode": None,
            "by_model": {},
        }

    def _normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        split = raw.get("test") or raw.get("val") or {}
        models = list(split.get("selected_models", []))
        weights = dict(split.get("weights", {}))
        protocol = raw.get("protocol", "protocol_b")
        relation_feedback = self._relation_feedback_audit(raw)
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
            "relation_feedback": relation_feedback,
            "predictions": predictions,
            "raw": raw,
        }


class CombinatorBackend(CombinationBackend):
    """Adapter around the legacy PowerModelCombinator.

    **迁移期兼容实现（Task 8 起）**：默认决策路径已切换到 `ProtocolBBackend`。
    本类仅在用户显式设置 `MODELCOMBINE_PIPELINE_BACKEND=combinator` 时启用，
    用途限于迁移期回退与历史对照。**禁止继续向旧引擎增加功能**；新的调度能力
    一律实现在 System B 主干上。计划在两个连续验收版本通过、并经用户再次确认后
    随 `PowerModelCombinator` 一并删除（Task 9）。

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
