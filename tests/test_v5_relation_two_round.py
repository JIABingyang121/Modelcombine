"""v5 关系强度闭环：同任务 warm-up 写边 -> 正式测量读边。

上一轮的接线仍有两个断点，导致"共享图"实际永远为空：

1. `_run_protocol_b_on_matrix` 构建 solver 时没有挂 temporal stage，
   于是图**只被读、不被写**，`edges_found` 恒为空。
2. 即便挂上，"前面任务写、后面任务读"这个假设也不成立——消费者按当前
   `scenario_id` 精确匹配，而九个任务的 dataset/horizon 不同、场景 ID 不同，
   PJM h=1 写的边 PJM h=6 根本读不到。

因此闭环必须落在**同一个任务**上：先 warm-up 产生当前场景→模型的
`recommended_for` 边，再用同一张已冻结的关系状态做正式测量。
测量阶段必须停止写入，否则任务的运行顺序会改变图，破坏可复现性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.run_system_ab_shadow as shadow
from src.core.solver import build_protocol_b_context
from src.eval.kg.feedback import KGFeedbackStore
from src.graph.model_graph import ModelGraph

DATASET = "pjm"
HORIZON = 1
MODELS = ["cand_a", "cand_b", "cand_c"]


def _frames(n: int = 600):
    ts_val = pd.date_range("2026-01-01", periods=n, freq="h")
    ts_test = pd.date_range("2026-04-01", periods=n // 3, freq="h")
    rng = np.random.default_rng(7)

    def mk(ts):
        m = len(ts)
        y = 100 + 20 * np.sin(np.arange(m) * 2 * np.pi / 24) + rng.normal(0, 1.0, m)
        d = {"timestamp": ts, "y": y}
        for i, name in enumerate(MODELS):
            d[name] = y + rng.normal(0, 1.0 + 0.02 * i, m)
        return pd.DataFrame(d)

    df_val, df_test = mk(ts_val), mk(ts_test)
    raw_val = pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour, "dow": ts_val.dayofweek})
    raw_test = pd.DataFrame({"timestamp": ts_test, "hour": ts_test.hour, "dow": ts_test.dayofweek})
    return df_val, df_test, raw_val, raw_test


def _matrix():
    df_val, df_test, raw_val, raw_test = _frames()
    return {
        "df_val_kg": df_val,
        "df_test_kg": df_test,
        "df_raw_val": raw_val,
        "df_raw_test": raw_test,
        "safe_models": list(MODELS),
        "base_model_cols": list(MODELS),
        "metadata": {
            "safe_models": list(MODELS),
            "common_base_models": list(MODELS),
            "eligible_filter_reasons": {},
            "filter": {},
            "frozen_naive": {"loaded": False},
            "raw": {"val_loaded": True, "test_loaded": True},
        },
    }


def _scenario_id(matrix) -> str:
    ctx = build_protocol_b_context(
        dataset=DATASET,
        horizon=HORIZON,
        df_val=matrix["df_val_kg"],
        df_test=matrix["df_test_kg"],
        df_raw_val=matrix["df_raw_val"],
        df_raw_test=matrix["df_raw_test"],
        model_cols=list(MODELS),
        base_model_cols=list(MODELS),
        feedback_store=None,
        return_predictions=True,
    )
    return ctx.scenario.scenario_id


def _run(matrix, graph, *, write_relations, trace_path=None):
    raw, _pred, _trace = shadow._run_protocol_b_on_matrix(
        dataset=DATASET,
        horizon=HORIZON,
        matrix=matrix,
        feedback_store=KGFeedbackStore(learning_rate=0.1),
        trace_path=trace_path,
        relation_graph=graph,
        write_relations=write_relations,
    )
    return raw


def _edges_found(raw):
    return list(
        (
            ((raw.get("val") or {}).get("weight_meta") or {})
            .get("protocol_b_selection_meta", {})
            .get("relation_strength", {})
        ).get("edges_found")
        or []
    )


def _scenario_out_edges(graph, scenario_id):
    if not graph.G.has_node(scenario_id):
        return []
    return [
        (tgt, data)
        for _s, tgt, data in graph.G.out_edges(scenario_id, data=True)
        if data.get("edge_type") == "recommended_for"
    ]


def test_warmup_writes_current_scenario_recommended_for_edges():
    """warm-up 必须真的把当前场景→模型的 recommended_for 边写进共享图。"""
    matrix = _matrix()
    graph = ModelGraph()
    _run(matrix, graph, write_relations=True)

    edges = _scenario_out_edges(graph, _scenario_id(matrix))
    assert edges, (
        "warm-up 后共享图里没有当前场景的 recommended_for 边——"
        "solver 未挂载 temporal stage，图只读不写"
    )
    for _tgt, data in edges:
        assert data.get("dynamic_strength") is not None


def test_second_round_consumes_edges_written_by_first_round():
    """真实两轮：第一轮写边，第二轮读到非空 edges_found。"""
    matrix = _matrix()
    graph = ModelGraph()

    first = _run(matrix, graph, write_relations=True)
    assert _edges_found(first) == [], "第一轮图为空，不应消费到任何关系边"

    second = _run(matrix, graph, write_relations=False)
    assert _edges_found(second), "第二轮未消费到第一轮写入的关系边，闭环仍未接通"


def test_measurement_run_must_not_mutate_the_shared_graph():
    """正式测量阶段不得写图，否则任务运行顺序会改变关系状态。"""
    matrix = _matrix()
    graph = ModelGraph()
    _run(matrix, graph, write_relations=True)
    before = (graph.G.number_of_nodes(), graph.G.number_of_edges())
    snapshot = {
        (s, t): dict(d) for s, t, d in graph.G.edges(data=True)
    }

    _run(matrix, graph, write_relations=False)

    assert (graph.G.number_of_nodes(), graph.G.number_of_edges()) == before
    for (s, t), old in snapshot.items():
        new = graph.G[s][t]
        assert new.get("dynamic_strength") == old.get("dynamic_strength")
        assert new.get("event_count") == old.get("event_count")


def test_run_task_records_relation_contrast_as_its_own_arm(tmp_path, monkeypatch):
    """v5 必须单独记录关系启用/中性两组结果。

    现有 `test_mae_on/off` 是 interaction 对照，两臂的关系强度是**一样的**，
    不能冒充关系强度收益对照。
    """
    matrix = _matrix()
    monkeypatch.setattr(shadow, "build_task_matrix", lambda **kwargs: matrix)

    graph = ModelGraph()
    record = shadow.run_task(
        dataset=DATASET,
        horizon=HORIZON,
        models=list(MODELS),
        pred_root=tmp_path,
        raw_root=None,
        out_root=tmp_path / "out",
        filter_threshold=0.0,
        seed=42,
        run_combinator=False,
        relation_graph=graph,
    )

    assert record["status"] == "ok", record.get("error")
    warmup = record.get("relation_warmup")
    assert warmup, "run_task 未做同任务 warm-up"
    assert warmup["edges_written"], "warm-up 未写入任何关系边"

    contrast = record.get("relation_contrast")
    assert contrast, "缺少关系启用/中性对照"
    assert contrast["enabled_edges_found"], "启用臂未消费到关系边"
    assert contrast["neutral_edges_found"] == [], "中性臂不得消费关系边"
    assert record["test_mae_relation_neutral"] is not None
    assert record["test_mae_relation_delta"] == pytest.approx(
        record["test_mae_on"] - record["test_mae_relation_neutral"]
    )
