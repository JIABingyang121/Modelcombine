"""关系强度在**真实生产路径**上被消费（不是函数级注入）。

上一轮 `97eaf3f` 只证明了 `select_models_protocol_b` 内部有效——测试直接把合成图
注入该函数。但真实链路是：

    ProtocolBBackend.combine -> kg_combination_with_features -> select_models_protocol_b

而 `backends.py` 调用引擎时**没有传 `ctx.model_graph`**，`protocol_b.py` 内部又
`mg = ModelGraph()` 新建空图，于是生产路径上恒为
`edges_found=[] / relation_strength=0.5 / contribution=0.0`。

这与本项目反复出现的"函数有效、真实接线空转"同类（树模型区间、Path 超边、
seasonal_naive 来源校验都栽在这里），故本模块的断言全部打在
`ProtocolBBackend.combine` 这一真实入口上。

另有一条口径要求：只能读**当前场景**指向模型的关系边。若把所有场景的边平均，
无关场景会改变当前决策。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.solver import build_protocol_b_context
from src.core.solver.backends import ProtocolBBackend
from src.core.trace import SelectionTrace
from src.graph.model_graph import ModelGraph

SCENARIO = "pjm"          # build_protocol_b_context 用 dataset 作为场景前缀
OTHER_SCENARIO = "other_scenario"
MODELS = ["cand_a", "cand_b", "cand_c"]


def _frames(n: int = 600):
    ts_val = pd.date_range("2026-01-01", periods=n, freq="h")
    ts_test = pd.date_range("2026-04-01", periods=n // 3, freq="h")
    rng = np.random.default_rng(11)

    def mk(ts):
        m = len(ts)
        y = 100 + 20 * np.sin(np.arange(m) * 2 * np.pi / 24) + rng.normal(0, 1.0, m)
        d = {"timestamp": ts, "y": y}
        for i, name in enumerate(MODELS):
            d[name] = y + rng.normal(0, 1.0 + 0.02 * i, m)
        return pd.DataFrame(d)

    df_val, df_test = mk(ts_val), mk(ts_test)
    raw_val = pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour, "dow": ts_val.dayofweek})
    raw_test = pd.DataFrame(
        {"timestamp": ts_test, "hour": ts_test.hour, "dow": ts_test.dayofweek}
    )
    return df_val, df_test, raw_val, raw_test


def _graph_with_scenario_relations(scenario_id: str, strengths: dict) -> ModelGraph:
    mg = ModelGraph()
    mg.add_scenario_node(scenario_id, {})
    for m in MODELS:
        mg.add_model_node(m, {})
    for m, w in strengths.items():
        mg.add_scenario_model_edge(scenario_id, m, weight=w)
    return mg


def _build_ctx():
    df_val, df_test, raw_val, raw_test = _frames()
    return build_protocol_b_context(
        dataset=SCENARIO,
        horizon=1,
        df_val=df_val,
        df_test=df_test,
        df_raw_val=raw_val,
        df_raw_test=raw_test,
        model_cols=list(MODELS),
        base_model_cols=list(MODELS),
        feedback_store=None,
    )


def current_scenario_id() -> str:
    """与 _run 完全同构地取当前场景 id。

    注意 scenario_id 由签名（含 n_val/n_test）派生：若构造探针上下文时用了
    不同的 df_test，会得到不同的 id、关系边永远匹配不上——首版测试正是栽在这里。
    """
    return _build_ctx().scenario.scenario_id


def _run(model_graph=None):
    ctx = _build_ctx()
    if model_graph is not None:
        ctx.model_graph = model_graph
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)
    ProtocolBBackend().combine(ctx, trace)
    stage = next(s for s in trace.stages if s["stage"] == "ProtocolBBackend")
    return stage["outputs"]


def _relation_of(outputs):
    rel = outputs.get("relation_strength")
    assert rel is not None, "trace 未记录 relation_strength"
    return rel


def test_production_path_consumes_current_scenario_relation_edges():
    """核心红灯：真实入口必须真的读到当前场景的关系边。"""
    graph = _graph_with_scenario_relations(
        current_scenario_id(), {"cand_a": 0.95, "cand_b": 0.05}
    )
    outputs = _run(model_graph=graph)

    rel = _relation_of(outputs)
    assert rel["edges_found"], (
        "生产路径未消费任何关系边——backends 未把 ctx.model_graph 传给引擎，"
        "或引擎内部新建了空图"
    )
    assert rel["by_model"]["cand_a"] == pytest.approx(0.95)
    assert rel["by_model"]["cand_b"] == pytest.approx(0.05)
    assert rel["contribution"]["cand_a"] > 0
    assert rel["contribution"]["cand_b"] < 0


def test_only_current_scenario_relations_are_used():
    """其他场景指向同一模型的关系边不得影响当前决策。"""
    graph = _graph_with_scenario_relations(current_scenario_id(), {"cand_a": 0.95})
    # 另一个场景对同一模型给出相反的强度，必须被忽略
    graph.add_scenario_node(OTHER_SCENARIO, {})
    graph.add_scenario_model_edge(OTHER_SCENARIO, "cand_a", weight=0.01)

    rel = _relation_of(_run(model_graph=graph))

    assert rel["by_model"]["cand_a"] == pytest.approx(0.95), (
        "读到了非当前场景的关系边（疑似对所有场景取平均）"
    )


def test_without_graph_relation_term_is_inert():
    """不传图谱时该项必须恒为中性，行为与接入前一致。"""
    rel = _relation_of(_run(model_graph=None))

    assert rel["edges_found"] == []
    for m in MODELS:
        assert rel["by_model"][m] == pytest.approx(0.5)
        assert rel["contribution"][m] == pytest.approx(0.0)


def test_trace_candidate_scores_and_ranking_come_from_real_engine_output():
    """候选得分与排序必须来自真实引擎产物结构，不能为空。

    真实引擎把 `model_scores_b` 写在 **val** 里（protocol_b.py 的最终返回与
    guarded_val 均如此），此前审计先读 test，导致生产路径上恒为空。
    """
    outputs = _run(model_graph=None)

    assert outputs["candidate_scores"], (
        "candidate_scores 为空——审计读取的 split 与真实引擎产物结构不一致"
    )
    assert set(outputs["candidate_scores"]) <= set(MODELS)
    assert outputs["candidate_ranking"], "candidate_ranking 为空"
    scores = outputs["candidate_scores"]
    assert outputs["candidate_ranking"] == sorted(
        outputs["candidate_ranking"], key=lambda m: -scores[m]
    ), "排序未按候选得分从高到低"
