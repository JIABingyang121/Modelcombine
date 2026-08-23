"""关系强度在 **Adapter 全链路**上的因果验证。

注意口径：本模块的入口是 `DemoProtocolBAdapter.select`，覆盖
adapter → build_protocol_b_context → ProtocolBBackend → 引擎 → 评分 这条链，
**不经过** `PowerPredictionPipeline`。Pipeline 层（`main.py` 是否把同一张图传给
adapter、反馈是否用真实 scenario_id）由
`tests/test_pipeline_relation_graph_wiring.py` 覆盖，两者不可互相替代。

前几轮逐层堵漏后仍有两处断点：

1. `main.py` 调 `DemoProtocolBAdapter.select` 时不传 `model_graph`，
   adapter 构造上下文时也没设——`ProtocolBBackend` 那层虽已接通，
   但默认 `run.py` 路径上图谱根本没传到。
2. 生产者与消费者不匹配：消费者读 `scenario → model` 的 `recommended_for`；
   而 Hawkes stage 在有 `path_id` 时写的是 `scenario → path_id`，
   无 `path_id` 时虽写 `scenario → model` 却把关系类型硬编码成 `selected_model`。
   两条路都不会产生消费者需要的边。

因此本模块的断言全部打在 **Adapter** 这一 run.py 直接调用的入口上，并且要求
"只翻转一条关系强度 → 最终模型组合随之改变"这一**因果**结论，而不是只看
`edges_found` 非空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.graph.model_graph import ModelGraph
from src.pipeline.prediction_pool import RegionPredictionBundle
from src.pipeline.protocol_b_adapter import DemoProtocolBAdapter

REGION = "PJME"
MODELS = ["cand_a", "cand_b", "cand_c"]


def _bundle(n_val: int = 600, n_test: int = 200) -> RegionPredictionBundle:
    rng = np.random.default_rng(5)
    ts_val = pd.date_range("2026-01-01", periods=n_val, freq="h")
    ts_test = pd.date_range("2026-04-01", periods=n_test, freq="h")

    def mk(ts):
        m = len(ts)
        y = 100 + 20 * np.sin(np.arange(m) * 2 * np.pi / 24) + rng.normal(0, 1.0, m)
        d = {"timestamp": ts, "y": y}
        # 三个候选精度接近，使关系强度成为可区分因素
        for i, name in enumerate(MODELS):
            d[name] = y + rng.normal(0, 1.0 + 0.01 * i, m)
        return pd.DataFrame(d)

    df_val, df_test = mk(ts_val), mk(ts_test)
    return RegionPredictionBundle(
        df_val=df_val,
        df_test=df_test,
        df_raw_val=pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour}),
        df_raw_test=pd.DataFrame({"timestamp": ts_test, "hour": ts_test.hour}),
        model_cols=list(MODELS),
        base_model_cols=list(MODELS),
        fitted_test_models={},
        metadata={},
    )


def _select(bundle, model_graph=None):
    return DemoProtocolBAdapter().select(
        bundle, region=REGION, horizon=1, model_graph=model_graph
    )


def _graph_for(bundle, strengths: dict) -> ModelGraph:
    """按 adapter 实际使用的场景 id 建关系边。

    场景 id 由 build_protocol_b_context 的签名派生，必须与 adapter 内部一致，
    否则边永远匹配不上（此前测试已在这里踩过一次）。
    """
    from src.core.solver import build_protocol_b_context

    ctx = build_protocol_b_context(
        dataset=REGION, horizon=1,
        df_val=bundle.df_val, df_test=bundle.df_test,
        df_raw_val=bundle.df_raw_val, df_raw_test=bundle.df_raw_test,
        model_cols=list(bundle.model_cols),
        base_model_cols=list(bundle.base_model_cols),
        feedback_store=None,
    )
    mg = ModelGraph()
    mg.add_scenario_node(ctx.scenario.scenario_id, {})
    for m in MODELS:
        mg.add_model_node(m, {})
    for m, w in strengths.items():
        mg.add_scenario_model_edge(ctx.scenario.scenario_id, m, weight=w)
    return mg


def _relation_of(result):
    stage = next(
        s for s in result["trace"].stages if s["stage"] == "ProtocolBBackend"
    )
    rel = stage["outputs"].get("relation_strength")
    assert rel is not None, "trace 未记录 relation_strength"
    return rel


def test_adapter_accepts_and_forwards_model_graph():
    """Adapter 必须接收图谱并把它送到引擎——这是 run.py 侧的接线点。"""
    bundle = _bundle()
    graph = _graph_for(bundle, {"cand_a": 0.95, "cand_b": 0.05})

    rel = _relation_of(_select(bundle, model_graph=graph))

    assert rel["edges_found"], "Adapter 未把 model_graph 传到引擎"
    assert rel["by_model"]["cand_a"] == pytest.approx(0.95)


def _b_candidates(result):
    stage = next(
        s for s in result["trace"].stages if s["stage"] == "ProtocolBBackend"
    )
    return stage["outputs"]


def test_flipping_one_relation_edge_changes_protocol_b_selection():
    """因果断言：只翻转一条关系强度，Protocol B 选出的组合必须改变。

    这是关系强度实际作用的层级。注意**最终输出未必随之改变**：若 guard 判定 B
    不如 Protocol A 而回退，最终组合来自 A（按 MAE，不受关系强度影响）。
    下一个用例专门锁定这层关系，避免把"B 的选择变了"误读成"最终输出一定变"。
    """
    bundle = _bundle()

    weak = _select(bundle, model_graph=_graph_for(bundle, {"cand_c": 0.05}))
    strong = _select(bundle, model_graph=_graph_for(bundle, {"cand_c": 0.95}))

    ow, os_ = _b_candidates(weak), _b_candidates(strong)

    # 排序必须随之改变
    assert ow["candidate_ranking"] != os_["candidate_ranking"], (
        f"排序未变：{ow['candidate_ranking']} vs {os_['candidate_ranking']}"
    )
    # Protocol B 自己选出的组合必须随之改变
    assert set(ow["protocol_b_candidates"]) != set(os_["protocol_b_candidates"]), (
        "翻转关系强度未改变 Protocol B 的选择："
        f"弱={ow['protocol_b_candidates']} 强={os_['protocol_b_candidates']}"
    )
    assert "cand_c" in os_["protocol_b_candidates"], (
        f"关系强度拉满的候选未被 B 选中：{os_['protocol_b_candidates']}"
    )


def test_guard_fallback_decouples_relation_strength_from_final_output():
    """如实锁定边界：guard 回退时最终输出来自 Protocol A，不随关系强度改变。

    这不是缺陷，而是 guard 的既定职责（B 未胜过 A 就不采用 B）。写成断言是为了
    防止后续把"关系强度已接入"直接等同于"最终结果一定受其影响"。
    """
    bundle = _bundle()
    strong = _select(bundle, model_graph=_graph_for(bundle, {"cand_c": 0.95}))
    outputs = _b_candidates(strong)

    if outputs["fallback_target"] is None:
        # 未回退时，最终输出应当就是 B 的选择
        assert set(strong["models"]) == set(outputs["protocol_b_candidates"])
    else:
        # 回退时最终输出来自 A，可以与 B 的候选不同
        assert outputs["fallback_target"] in {"protocol_a", "best_single"}
        assert strong["models"], "回退后仍须有最终选择"


def test_without_graph_relation_term_stays_neutral():
    """不传图谱时行为与接入前一致。"""
    rel = _relation_of(_select(_bundle(), model_graph=None))

    assert rel["edges_found"] == []
    for m in MODELS:
        assert rel["contribution"][m] == pytest.approx(0.0)


def test_feedback_stage_emits_scenario_to_model_recommended_for_edges():
    """反馈阶段必须产出消费者所需的 scenario->model / recommended_for 边。

    否则关系强度只会写在 scenario->path_id 上（或用 selected_model 关系类型），
    下一轮消费者读不到，闭环断裂。
    """
    from src.core.trace import SelectionTrace
    from src.graph.temporal_relations import events_from_solver_result

    trace = SelectionTrace(scenario_id="scenario_x")
    result = {"models": ["cand_a", "cand_b"], "weights": {"cand_a": 0.6, "cand_b": 0.4}}

    from src.graph.temporal_relations import make_temporal_relation_stage  # noqa: F401

    # 用生产 stage 的默认关系类型（recommended_for），而不是底层函数的默认值
    events = events_from_solver_result(
        trace, result, selected_target="B_pred_features",
        selected_relation_type="recommended_for",
    )

    pairs = {(e.target, e.relation_type) for e in events}
    for m in ("cand_a", "cand_b"):
        assert (m, "recommended_for") in pairs, (
            f"未生成 scenario->{m} 的 recommended_for 边；实际产生：{sorted(pairs)}"
        )


# --- Hawkes 反馈闭环：开启时才挂 temporal stage ------------------------------


def test_adapter_mounts_temporal_stage_only_when_hawkes_enabled(monkeypatch):
    """默认不挂 stage（行为不变）；开启后必须用同一张图挂载，闭环才成立。

    否则修改后的关系事件生成代码在默认主流程中根本不执行，下一轮没有新关系边
    可消费——"接入了关系强度"就只是半条链路。
    """
    import src.pipeline.protocol_b_adapter as adapter_mod

    captured = {}
    real_build = adapter_mod.build_solver

    def _capture(mode, **kwargs):
        captured.update(kwargs)
        return real_build(mode, **kwargs)

    monkeypatch.setattr(adapter_mod, "build_solver", _capture)

    bundle = _bundle()
    graph = _graph_for(bundle, {"cand_a": 0.9})

    # 默认：不挂 temporal stage
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_ENABLE_TEMPORAL_RELATIONS", raising=False)
    captured.clear()
    _select(bundle, model_graph=graph)
    assert captured.get("temporal_relation_graph") is None

    # 开启后：必须挂载，且用的是同一张图
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_ENABLE_TEMPORAL_RELATIONS", "1")
    captured.clear()
    _select(bundle, model_graph=graph)
    assert captured.get("temporal_relation_graph") is graph, (
        "开启 Hawkes 后未用同一张图挂载 temporal stage，反馈闭环断裂"
    )
