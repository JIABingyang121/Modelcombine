"""reasoning 收敛为有来源的逐模型评分证据（Task 8.3 Task 2）。

冷启动路径（无历史推荐边）必须输出零贡献，不得参与模型取舍；只有带
`evidence_refs` 的历史路径才产生非中性贡献。贡献在 base_scores 形成之前计算，
作为逐模型先验交给唯一候选选择器。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.solver.backends import ProtocolBBackend
from src.core.solver.protocol_b_context import build_protocol_b_context
from src.core.trace import SelectionTrace
from src.eval.kg.reasoning_evidence import build_reasoning_evidence
from src.graph.model_graph import ModelGraph


def _reasoning_graph(*, with_history: bool) -> ModelGraph:
    graph = ModelGraph()
    graph.add_scenario_node("s", {})
    graph.add_feature_node("f")
    for model in ("m1", "m2"):
        graph.add_model_node(model, {"input_constraints": {"features": ["f"]}})
        graph.add_feature_model_edge("f", model)
    graph.add_scenario_feature_edge("s", "f")
    if with_history:
        path_id = graph.instantiate_path(
            "path::m2", ["m2"], "single", evidence_ref="ev::path"
        )
        graph.add_scenario_path_edge(
            "s", path_id, performance_score=0.8, evidence_ref="ev::history"
        )
    return graph


def test_cold_start_paths_have_zero_model_contribution():
    evidence = build_reasoning_evidence(
        graph=_reasoning_graph(with_history=False), scenario_id="s", available_features={"f"},
        model_cols=["m1", "m2"], max_models=2, mode="hybrid",
    )
    assert evidence.source == "cold_start_no_evidence"
    assert evidence.contribution_by_model == {"m1": 0.0, "m2": 0.0}


def test_historical_path_contributes_before_selection():
    evidence = build_reasoning_evidence(
        graph=_reasoning_graph(with_history=True), scenario_id="s", available_features={"f"},
        model_cols=["m1", "m2"], max_models=2, mode="hybrid",
    )
    assert evidence.source == "historical_evidence"
    assert evidence.contribution_by_model["m2"] > 0
    assert "ev::history" in evidence.paths[0]["evidence_refs"]


def test_real_engine_records_reasoning_evidence_and_never_rewrites_stepwise():
    """真实引擎路径：reasoning 证据必须写入 selection_flow，且不再覆盖 stepwise。

    冷启动路径（真实数据 warm-up 只写 recommended_for 边，不写历史路径）下
    reasoning 证据恒为零贡献，但不能因此空转——证据来源与贡献必须出现在 trace，
    且 `after_reasoning` 必须等于 `after_stepwise`。
    """
    models = ["m1", "m2", "m3"]
    rng = np.random.default_rng(42)
    ts_val = pd.date_range("2026-01-01", periods=600, freq="h")
    ts_test = pd.date_range("2026-02-01", periods=180, freq="h")

    def make(ts):
        m = len(ts)
        y = 100.0 + 10.0 * np.sin(np.arange(m) * 2 * np.pi / 24)
        data = {"timestamp": ts, "y": y}
        for i, name in enumerate(models):
            data[name] = y + rng.normal(0.0, 1.0 + 0.02 * i, m)
        return pd.DataFrame(data)

    df_val, df_test = make(ts_val), make(ts_test)
    ctx = build_protocol_b_context(
        dataset="pjm",
        horizon=1,
        df_val=df_val,
        df_test=df_test,
        df_raw_val=pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour}),
        df_raw_test=pd.DataFrame({"timestamp": ts_test, "hour": ts_test.hour}),
        model_cols=list(models),
        base_model_cols=list(models),
        feedback_store=None,
    )
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)
    ProtocolBBackend().combine(ctx, trace)
    outputs = next(s for s in trace.stages if s["stage"] == "ProtocolBBackend")["outputs"]
    flow = outputs["selection_flow"]

    assert flow["reasoning"]["source"] == "cold_start_no_evidence"
    contribution = flow["reasoning"]["contribution"]
    assert isinstance(contribution, dict) and contribution
    assert all(v == 0.0 for v in contribution.values())
    assert flow["post_selector_mutations"] == []
    assert flow["selector_output"] == flow["final_selected_before_fit"]
