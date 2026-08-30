"""候选资格与选择目标对齐（Task 8.3 Task 10）。

Protocol B 不得把"权重归零后已退化为单模型"的 pair 当作有效二模型组合：
资格判定复用生产 zero-weight cleanup 的 `after`，替换候选只用 validation 数据，
找不到合格 pair 时必须显式回退并留下原因。stepwise 返回少于两个模型而未被采用时，
trace 必须说明未采用原因和后续实际使用的候选来源。

本文件的退化全部由真实 Ridge + 真实 cleanup 产生，不 monkeypatch 资格字段。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.enums import TaskType
from src.core.schema import DataContract, ScenarioDefinition
from src.core.solver.backends import ProtocolBBackend
from src.core.solver.context import SolveContext
from src.core.trace import SelectionTrace
from src.eval.kg.config import KG_ZERO_WEIGHT_CLEANUP_THRESHOLD
from src.eval.kg.protocol_a import kg_combination_pred_only
from src.eval.kg.protocol_b import kg_combination_with_features
from tests.task83_fixtures import (
    ALL_DEGENERATE_MODELS,
    DEGENERATE_MODELS,
    MODELS,
    make_all_degenerate_frames,
    make_degenerate_pair_frames,
    make_protocol_b_frames,
)


def _scenario():
    return ScenarioDefinition(
        task_type=TaskType.FORECASTING,
        business_domain="load_forecast",
        data_contract=DataContract(
            required_columns={"load": "float"},
            freq="H",
            min_samples=5,
            business_domain="load_forecast",
        ),
        target_schema={"yhat": "float"},
        primary_metric="MAE",
        signature_features=["load"],
        signature={"load": 1.0},
        region="R",
    )


def _run_b(frames, model_cols, horizon=1):
    df_val, df_test, raw_val, raw_test = frames
    return kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, list(model_cols), horizon,
        dataset_name="pjm", base_model_cols=list(model_cols),
    )


def _flow(result):
    return result["val"]["weight_meta"]["protocol_b_selection_meta"]["selection_flow"]


def test_degenerate_stepwise_pair_is_replaced_by_eligible_pair():
    """stepwise 选中的 pair 权重塌缩时，必须换成 validation 上最好的合格 pair。"""
    result = _run_b(make_degenerate_pair_frames(), DEGENERATE_MODELS)
    flow = _flow(result)
    eligibility = flow["pair_eligibility"]

    # 生产 stepwise 仍然先选中退化 pair，资格判定必须把它记成不合格
    assert eligibility["incumbent_pair"] == ["m_anchor", "m_scaled"]
    incumbent = [
        row for row in eligibility["evaluated_pairs"]
        if row["models"] == ["m_anchor", "m_scaled"]
    ]
    assert len(incumbent) == 1
    assert incumbent[0]["eligible_pair"] is False
    assert incumbent[0]["degenerate_reason"] == "zero_weight_cleanup_reduced_pair"
    assert incumbent[0]["effective_models"] == ["m_scaled"]

    # 最终不得继续使用退化 pair；换成 validation MAE 最低的合格 pair
    assert flow["selector_output"] != ["m_anchor", "m_scaled"]
    assert eligibility["outcome"] == "replaced_by_full_validation"
    assert eligibility["selected_pair"] == ["m_third", "m_scaled"]
    assert flow["selector_output"] == ["m_third", "m_scaled"]

    eligible_rows = [row for row in eligibility["evaluated_pairs"] if row["eligible_pair"]]
    assert eligible_rows, "至少要有一条合格 pair 才谈得上替换"
    best = min(eligible_rows, key=lambda row: row["validation_mae"])
    assert best["models"] == eligibility["selected_pair"]
    # 退化 pair 并不是因为 MAE 差才被淘汰：它在 validation 上仍好于部分合格 pair，
    # 说明资格判定确实先于 MAE 排序执行
    worst_eligible = max(eligible_rows, key=lambda row: row["validation_mae"])
    assert incumbent[0]["validation_mae"] < worst_eligible["validation_mae"]

    # 换过之后拟合出来的两个权重都必须是真实非零权重
    weights = result["val"]["weights"]
    assert set(weights) == {"m_third", "m_scaled"}
    assert all(abs(w) > KG_ZERO_WEIGHT_CLEANUP_THRESHOLD for w in weights.values())

    # 换候选走的是既有 constraint_decisions 通道，不另起一套记录
    stages = [d["stage"] for d in flow["constraint_decisions"]]
    assert "full_validation_pair_selection" in stages


def test_pair_selection_does_not_read_test_labels_or_test_predictions():
    """改掉 test 标签与 test 预测都不得改变 pair 选择。

    df_test 的模型列按固定排列重排：逐列分布不变（PSI/drift 判定不受影响），
    但 y 与预测的配对被打乱，test MAE 大幅变化。若选择读了 test，结果必然改变。
    """
    df_val, df_test, raw_val, raw_test = make_degenerate_pair_frames()
    baseline = _run_b((df_val, df_test, raw_val, raw_test), DEGENERATE_MODELS)

    mutated = df_test.copy()
    order = np.random.default_rng(99).permutation(len(mutated))
    for col in DEGENERATE_MODELS:
        mutated[col] = mutated[col].values[order]
    mutated["y"] = mutated["y"].values * 3.0 + 500.0
    perturbed = _run_b((df_val, mutated, raw_val, raw_test), DEGENERATE_MODELS)

    # 前提核对：drift 判据来自 test 预测分布，重排后必须仍然一致
    assert (
        baseline["val"]["weight_meta"]["protocol_b_selection_meta"]["drift_level"]
        == perturbed["val"]["weight_meta"]["protocol_b_selection_meta"]["drift_level"]
    )
    assert baseline["test"]["mae"] != pytest.approx(perturbed["test"]["mae"])

    base_flow, pert_flow = _flow(baseline), _flow(perturbed)
    assert base_flow["selector_output"] == pert_flow["selector_output"]
    assert base_flow["pair_eligibility"] == pert_flow["pair_eligibility"]


def test_all_pairs_degenerate_falls_back_with_recorded_reason():
    """没有任何合格 pair 时，必须显式回退到清理后的有效模型并记录原因。"""
    result = _run_b(make_all_degenerate_frames(), ALL_DEGENERATE_MODELS)
    flow = _flow(result)
    eligibility = flow["pair_eligibility"]

    assert eligibility["evaluated_pairs"], "必须留下被评估的 pair"
    assert all(row["eligible_pair"] is False for row in eligibility["evaluated_pairs"])
    assert all(
        row["degenerate_reason"] == "zero_weight_cleanup_reduced_pair"
        for row in eligibility["evaluated_pairs"]
    )
    assert eligibility["outcome"] == "no_eligible_pair"
    assert eligibility["selected_pair"] is None
    assert eligibility["fallback_reason"] == "no_eligible_pair_after_zero_weight_cleanup"
    assert eligibility["fallback_target"] == ["c1"]

    # 不得静默保留退化 pair
    assert flow["selector_output"] == ["c1"]
    stages = [d["stage"] for d in flow["constraint_decisions"]]
    assert "degenerate_pair_fallback_to_effective_single" in stages

    # selector 输出正确还不够：顶层最终输出同样不能带回退化 pair
    assert result["val"]["selected_models"] != ["c1", "c2"]
    assert all(
        abs(w) > KG_ZERO_WEIGHT_CLEANUP_THRESHOLD
        for w in result["val"]["weights"].values()
    )


def test_no_eligible_pair_final_output_falls_back_to_best_single():
    """无合格 pair 时，顶层最终输出必须是最佳单模型，不得经 Protocol A 带回退化 pair。

    这条直接断言 `result["val"/"test"]["selected_models"]` 与最终权重：只看
    `selection_flow.selector_output` 会漏掉 guard 之后的回退目标。
    """
    result = _run_b(make_all_degenerate_frames(), ALL_DEGENERATE_MODELS)

    assert result["protocol"] == "B_fallback_to_best_single_guard"
    for split in ("val", "test"):
        assert result[split]["selected_models"] == ["c1"]
        assert result[split]["weights"] == {"c1": 1.0}

    guard_cfg = result["val"]["weight_meta"]["guard_config"]
    assert guard_cfg["final_fallback_target"] == "best_single"
    assert "no_eligible_pair_fallback_to_best_single" in guard_cfg["final_fallback_reason"]

    # 回退之后仍要能说明"B 本来想选什么、为什么不合格"
    flow = _flow(result)
    assert flow["pair_eligibility"]["outcome"] == "no_eligible_pair"
    assert result["val"]["selected_models_b_candidate"] == ["c1"]


def test_no_eligible_pair_fallback_is_recorded_in_selection_trace():
    """经 ProtocolBBackend 走一遍：SelectionTrace 必须说明这次回退。"""
    df_val, df_test, raw_val, raw_test = make_all_degenerate_frames()
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"hour"},
        model_cols=list(ALL_DEGENERATE_MODELS),
        df_val=df_val,
        df_test=df_test,
        df_raw_val=raw_val,
        df_raw_test=raw_test,
        horizon=1,
        dataset_name="pjm",
        base_model_cols=list(ALL_DEGENERATE_MODELS),
    )
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)
    result = ProtocolBBackend().combine(ctx, trace)

    assert result["models"] == ["c1"]
    stage = next(s for s in trace.stages if s["stage"] == "ProtocolBBackend")
    outputs = stage["outputs"]
    assert outputs["protocol"] == "B_fallback_to_best_single_guard"
    assert outputs["fallback_target"] == "best_single"
    assert "no_eligible_pair_fallback_to_best_single" in outputs["fallback_reason"]
    assert outputs["protocol_b_candidates"] == ["c1"]
    assert outputs["selection_flow"]["pair_eligibility"]["outcome"] == "no_eligible_pair"


def test_stepwise_shorter_than_two_records_non_adoption_and_candidate_source():
    """stepwise 只返回 1 个模型时，trace 必须写明未采用原因和实际候选来源。"""
    result = _run_b(make_all_degenerate_frames(), ALL_DEGENERATE_MODELS)
    adoption = _flow(result)["stepwise_adoption"]

    assert adoption["stepwise_output"] == ["c1"]
    assert adoption["adopted"] is False
    assert adoption["not_adopted_reason"] == "stepwise_returned_fewer_than_two_models"
    assert adoption["candidate_source"] == "beam_search"


def test_non_degenerate_input_keeps_previous_selection():
    """没有退化 pair 的普通输入行为不变：选择、权重、stepwise 采用记录都保持原样。"""
    result = _run_b(make_protocol_b_frames(), MODELS)
    flow = _flow(result)

    assert flow["selector_output"] == ["m3", "m2"]
    assert flow["fitted_models"] == ["m3", "m2"]
    assert result["val"]["weights"]["m3"] == pytest.approx(0.53096, rel=1e-4)
    assert result["val"]["weights"]["m2"] == pytest.approx(0.468587, rel=1e-4)

    adoption = flow["stepwise_adoption"]
    assert adoption["adopted"] is True
    assert adoption["stepwise_output"] == ["m3", "m2"]
    assert adoption["candidate_source"] == "stepwise"

    eligibility = flow["pair_eligibility"]
    assert eligibility["outcome"] == "kept_by_full_validation"
    assert eligibility["selected_pair"] == ["m3", "m2"]
    assert {frozenset(row["models"]) for row in eligibility["evaluated_pairs"]} == {
        frozenset(pair) for pair in [("m1", "m2"), ("m1", "m3"), ("m2", "m3")]
    }
    assert all(row["eligible_pair"] is True for row in eligibility["evaluated_pairs"])
    stages = [d["stage"] for d in flow["constraint_decisions"]]
    assert "degenerate_pair_replaced_by_eligible_pair" not in stages
    assert "degenerate_pair_fallback_to_effective_single" not in stages


@pytest.mark.parametrize(
    "frames_factory, model_cols, expected_models, expected_weights, expected_val_mae, expected_trace",
    [
        (
            make_degenerate_pair_frames,
            DEGENERATE_MODELS,
            ["m_scaled"],
            {"m_scaled": 1.0},
            15.040953309320912,
            [
                {"step": 0, "selected": "m_anchor", "mae": 0.7447123705302574},
                {
                    "step": 1, "selected": "m_scaled",
                    "mae": 0.3615238564750557, "rel_improve": 0.5145456544281117,
                },
            ],
        ),
        (
            make_protocol_b_frames,
            MODELS,
            ["m1", "m2"],
            {"m1": 0.5172847920256982, "m2": 0.4823541876594839},
            0.5449096964841889,
            [
                {"step": 0, "selected": "m1", "mae": 0.774347449034785},
                {
                    "step": 1, "selected": "m2",
                    "mae": 0.5320016888041754, "rel_improve": 0.3129677259642177,
                },
            ],
        ),
    ],
    ids=["degenerate_frames", "normal_frames"],
)
def test_protocol_a_selection_and_ordering_unchanged(
    frames_factory, model_cols, expected_models, expected_weights,
    expected_val_mae, expected_trace,
):
    """Protocol A 的逐步排序、最终选择与权重必须与本次改动前逐位一致。"""
    df_val, df_test, _raw_val, _raw_test = frames_factory()
    result = kg_combination_pred_only(
        df_val, df_test, list(model_cols), 1, 0.5, dataset_name="pjm",
    )
    assert result["val"]["selected_models"] == expected_models
    for name, weight in expected_weights.items():
        assert result["val"]["weights"][name] == pytest.approx(weight, rel=1e-9)
    assert result["val"]["mae"] == pytest.approx(expected_val_mae, rel=1e-9)

    trace = result["val"]["weight_meta"]["selection_meta"]["stepwise"]["trace"]
    assert len(trace) == len(expected_trace)
    for actual, expected in zip(trace, expected_trace):
        assert actual["selected"] == expected["selected"]
        assert actual["step"] == expected["step"]
        assert actual["mae"] == pytest.approx(expected["mae"], rel=1e-9)
