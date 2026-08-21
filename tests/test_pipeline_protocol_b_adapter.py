"""DemoProtocolBAdapter 契约测试（System A/B 合一 Task 3）。

Adapter 把 Task 2 的 `RegionPredictionBundle` 喂给统一 solver 的
`ProtocolBBackend`，并把 Protocol B 的结果翻译成 demo 侧统一的
`models/weights/strategy/path_id/yhat` + trace。

**一个必须显式处理的事实**：Protocol B 只返回指标，不返回预测向量，demo 侧的
yhat 只能由 `df_test[selected] @ weights` 重算。交互残差分支被接受时
（`weight_meta.interaction_branch.applied=True`）其 pred_test 不再是线性组合；
但随后的 post_adjustment 若被接受，又会用线性组合整体覆盖回去。因此不一致只在
"交互应用且 post_adjustment 未应用"时出现——源码上可达，实测未复现。Adapter
因此必须重算后核对 MAE 并在不一致时显式标记，而不是放宽容差假装一致。

注：下方用 monkeypatch 替换 Protocol B 引擎，测的是 adapter 自身契约。所用的
元数据键名均取自 protocol_b.py 真实产出并经真实引擎跑通核对过（早期版本曾用
臆造键名 `protocol_b_interaction`，导致该用例空转，已修正）。
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.protocol_b_adapter import DemoProtocolBAdapter

REGION = "R1"


def _bundle(n_test=12, n_val=10):
    from src.pipeline.prediction_pool import RegionPredictionBundle

    ts_val = pd.date_range("2026-01-01", periods=n_val, freq="h")
    ts_test = pd.date_range("2026-02-01", periods=n_test, freq="h")
    rng = np.random.default_rng(7)
    y_test = np.linspace(100.0, 120.0, n_test)
    df_val = pd.DataFrame(
        {
            "timestamp": ts_val,
            "y": np.linspace(90.0, 100.0, n_val),
            "m1": np.linspace(90.5, 100.5, n_val),
            "m2": np.linspace(89.0, 99.0, n_val),
        }
    )
    df_test = pd.DataFrame(
        {
            "timestamp": ts_test,
            "y": y_test,
            "m1": y_test + rng.normal(0, 1.0, n_test),
            "m2": y_test + rng.normal(0, 2.0, n_test),
        }
    )
    raw_val = pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour, "dow": ts_val.dayofweek})
    raw_test = pd.DataFrame({"timestamp": ts_test, "hour": ts_test.hour, "dow": ts_test.dayofweek})
    return RegionPredictionBundle(
        df_val=df_val,
        df_test=df_test,
        df_raw_val=raw_val,
        df_raw_test=raw_test,
        model_cols=["m1", "m2"],
        base_model_cols=["m1"],
        fitted_test_models={},
        metadata={"region": REGION},
    )


def _fake_raw(bundle, *, selected, weights, protocol="B_pred_features", extra_test=None):
    """构造 Protocol B 风格的返回值；mae 按真实加权组合算，保证自洽。"""
    yhat = bundle.df_test[selected].values @ np.array([weights[m] for m in selected], dtype=float)
    y = bundle.df_test["y"].values
    test_payload = {
        "mae": float(np.mean(np.abs(yhat - y))),
        "selected_models": list(selected),
        "weights": dict(weights),
    }
    if extra_test:
        test_payload.update(extra_test)
    return {
        "val": {"mae": 1.0, "selected_models": list(selected), "weights": dict(weights)},
        "test": test_payload,
        "protocol": protocol,
    }


def _patch_protocol_b(monkeypatch, raw):
    """替换 Protocol B 引擎本体；本任务测的是 adapter 契约，不是 KG 引擎内部。"""
    import src.eval.kg.protocol_b as pb

    monkeypatch.setattr(pb, "kg_combination_with_features", lambda *a, **k: raw)
    return raw


# --- Step 1: adapter 输入输出契约 --------------------------------------------


def test_adapter_returns_unified_fields_and_trace(monkeypatch, tmp_path):
    bundle = _bundle()
    _patch_protocol_b(monkeypatch, _fake_raw(bundle, selected=["m1"], weights={"m1": 1.0}))

    result = DemoProtocolBAdapter().select(
        bundle, region=REGION, horizon=1, trace_path=tmp_path / "t.json"
    )

    assert result["models"] == ["m1"]
    assert result["weights"] == {"m1": 1.0}
    assert result["strategy"] == "B_pred_features"
    assert result["path_id"] == "B_pred_features"
    assert len(result["yhat"]) == len(bundle.df_test)
    assert result["trace"] is not None
    assert (tmp_path / "t.json").exists()


def test_yhat_is_weighted_combination_of_selected_models(monkeypatch, tmp_path):
    bundle = _bundle()
    weights = {"m1": 0.7, "m2": 0.3}
    _patch_protocol_b(monkeypatch, _fake_raw(bundle, selected=["m1", "m2"], weights=weights))

    result = DemoProtocolBAdapter().select(bundle, region=REGION, horizon=1)

    expected = bundle.df_test["m1"].values * 0.7 + bundle.df_test["m2"].values * 0.3
    np.testing.assert_allclose(result["yhat"], expected, rtol=0, atol=1e-12)


def test_recomputed_mae_matches_protocol_b_within_tolerance(monkeypatch, tmp_path):
    bundle = _bundle()
    weights = {"m1": 0.6, "m2": 0.4}
    raw = _patch_protocol_b(
        monkeypatch, _fake_raw(bundle, selected=["m1", "m2"], weights=weights)
    )

    result = DemoProtocolBAdapter().select(bundle, region=REGION, horizon=1)

    assert result["mae_matches_protocol_b"] is True
    assert result["linear_reconstruction_match"] is True
    assert abs(result["mae"] - raw["test"]["mae"]) < 1e-8


def test_guard_fallback_to_best_single_still_reconciles(monkeypatch, tmp_path):
    """guard 回退到最优单模型时，权重是 {model: 1.0}，重算必须仍然一致。"""
    bundle = _bundle()
    raw = _patch_protocol_b(
        monkeypatch,
        _fake_raw(
            bundle,
            selected=["m2"],
            weights={"m2": 1.0},
            protocol="B_fallback_to_best_single_guard",
            extra_test={"selected_models_b_candidate": ["m1", "m2"]},
        ),
    )

    result = DemoProtocolBAdapter().select(bundle, region=REGION, horizon=1)

    assert result["strategy"] == "B_fallback_to_best_single_guard"
    assert result["models"] == ["m2"]
    assert result["mae_matches_protocol_b"] is True
    np.testing.assert_allclose(result["yhat"], bundle.df_test["m2"].values, atol=1e-12)


def test_linear_reconstruction_mismatch_is_surfaced_when_engine_preds_absent(monkeypatch, tmp_path):
    """引擎未交出预测时只能线性重建；不一致必须显式标记，且不得报假 MAE。"""
    bundle = _bundle()
    raw = _fake_raw(bundle, selected=["m1"], weights={"m1": 1.0})
    raw["test"]["mae"] = raw["test"]["mae"] + 5.0  # 模拟 pred_test 被交互残差修改
    raw["test"]["weight_meta"] = {"interaction_branch": {"applied": True}}
    _patch_protocol_b(monkeypatch, raw)

    result = DemoProtocolBAdapter().select(bundle, region=REGION, horizon=1)

    assert result["yhat_source"] == "linear_reconstruction"
    assert result["linear_reconstruction_match"] is False
    # 报的 MAE 必须是 yhat 真实算出来的，不能照抄引擎那个对不上的值
    assert result["mae"] == pytest.approx(
        float(np.mean(np.abs(result["yhat"] - bundle.df_test["y"].values)))
    )
    assert result["mae_matches_protocol_b"] is False
    stage = _find_stage(result["trace"], "DemoProtocolBAdapter")
    assert stage["outputs"]["linear_reconstruction_match"] is False
    assert stage["outputs"]["reconcile_note"].startswith("interaction_branch_applied")


def test_empty_selection_raises(monkeypatch, tmp_path):
    bundle = _bundle()
    _patch_protocol_b(monkeypatch, _fake_raw(bundle, selected=[], weights={}))

    with pytest.raises(ValueError, match="selected no model"):
        DemoProtocolBAdapter().select(bundle, region=REGION, horizon=1)


def test_feedback_store_is_isolated_per_region(monkeypatch, tmp_path):
    """初期不接生产反馈：每次 select 使用隔离实例，不写共享存储。"""
    bundle = _bundle()
    _patch_protocol_b(monkeypatch, _fake_raw(bundle, selected=["m1"], weights={"m1": 1.0}))

    adapter = DemoProtocolBAdapter()
    # 两个结果都保持引用存活后再比较同一性——早期版本用 id() 比较，
    # 第一个 store 被回收后地址会被复用，导致该断言时真时假（曾侥幸通过）。
    r1 = adapter.select(bundle, region="RA", horizon=1)
    r2 = adapter.select(bundle, region="RB", horizon=1)

    assert r1["feedback_store"] is not None
    assert r1["feedback_store"] is not r2["feedback_store"]


# --- Step 2: ProtocolBBackend 的 trace 增强 -----------------------------------


def _find_stage(trace, name):
    for stage in trace.stages:
        if stage["stage"] == name:
            return stage
    raise AssertionError(f"stage {name} not found in {[s['stage'] for s in trace.stages]}")


def test_backend_trace_records_b_candidates_and_fallback_target(monkeypatch, tmp_path):
    bundle = _bundle()
    raw = _fake_raw(
        bundle,
        selected=["m2"],
        weights={"m2": 1.0},
        protocol="B_fallback_to_best_single_guard",
        extra_test={
            "selected_models_b_candidate": ["m1", "m2"],
            "weights_b_candidate": {"m1": 0.5, "m2": 0.5},
            "weight_meta": {
                "protocol_b_guard": {
                    "fallback_target": "best_single",
                    "reason": "val_guard;degradation",
                },
            },
        },
    )
    _patch_protocol_b(monkeypatch, raw)

    result = DemoProtocolBAdapter().select(bundle, region=REGION, horizon=1)

    stage = _find_stage(result["trace"], "ProtocolBBackend")
    outputs = stage["outputs"]
    assert outputs["protocol_b_candidates"] == ["m1", "m2"]
    assert outputs["fallback_target"] == "best_single"
    assert "val_guard" in outputs["fallback_reason"]
    # 不再只记录最终模型列表
    assert outputs["models"] == ["m2"]


def test_backend_trace_records_guard_removed_models_as_rejections(monkeypatch, tmp_path):
    bundle = _bundle()
    raw = _fake_raw(
        bundle,
        selected=["m1"],
        weights={"m1": 1.0},
        extra_test={
            "weight_meta": {
                "protocol_b_selection_meta": {
                    "stability": {"removed_models": ["m2"]},
                },
            },
        },
    )
    _patch_protocol_b(monkeypatch, raw)

    result = DemoProtocolBAdapter().select(bundle, region=REGION, horizon=1)

    assert "m2" in result["trace"].candidates_rejected
    assert "stability" in result["trace"].candidates_rejected["m2"]


# --- Step 3/4: 三态后端开关与影子对照 ----------------------------------------


def test_backend_mode_defaults_to_combinator(monkeypatch):
    from src.pipeline.main import resolve_backend_mode

    monkeypatch.delenv("MODELCOMBINE_PIPELINE_BACKEND", raising=False)

    assert resolve_backend_mode() == "combinator"


@pytest.mark.parametrize("mode", ["combinator", "protocol_b_shadow", "protocol_b"])
def test_backend_mode_accepts_three_known_values(monkeypatch, mode):
    from src.pipeline.main import resolve_backend_mode

    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", mode)

    assert resolve_backend_mode() == mode


def test_unknown_backend_mode_raises_instead_of_silent_fallback(monkeypatch):
    from src.pipeline.main import resolve_backend_mode

    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocolb")

    with pytest.raises(ValueError) as exc:
        resolve_backend_mode()
    # 报错必须指名合法取值，且不得悄悄回退到 combinator
    assert "protocolb" in str(exc.value)
    assert "combinator" in str(exc.value)


def test_pipeline_exposes_resolved_backend_mode(monkeypatch):
    from src.pipeline.main import PowerPredictionPipeline

    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    pipeline = PowerPredictionPipeline.__new__(PowerPredictionPipeline)

    assert pipeline.backend_mode == "protocol_b_shadow"


def _results():
    combinator = {
        "models": ["m1"],
        "weights": {"m1": 1.0},
        "yhat": np.array([1.0, 2.0, 3.0]),
        "mae_recomputed": 0.5,
    }
    protocol_b = {
        "models": ["m1", "m2"],
        "weights": {"m1": 0.5, "m2": 0.5},
        "yhat": np.array([9.0, 9.0, 9.0]),
        "mae_recomputed": 0.4,
    }
    return combinator, protocol_b


def test_shadow_mode_final_output_stays_combinator():
    from src.pipeline.protocol_b_adapter import select_final_output

    combinator, protocol_b = _results()

    chosen = select_final_output("protocol_b_shadow", combinator, protocol_b)

    assert chosen is combinator
    np.testing.assert_allclose(chosen["yhat"], combinator["yhat"])


def test_protocol_b_mode_final_output_is_protocol_b():
    from src.pipeline.protocol_b_adapter import select_final_output

    combinator, protocol_b = _results()

    assert select_final_output("protocol_b", combinator, protocol_b) is protocol_b


def test_combinator_mode_ignores_protocol_b_result():
    from src.pipeline.protocol_b_adapter import select_final_output

    combinator, protocol_b = _results()

    assert select_final_output("combinator", combinator, None) is combinator


def test_select_final_output_rejects_unknown_mode():
    from src.pipeline.protocol_b_adapter import select_final_output

    combinator, protocol_b = _results()

    with pytest.raises(ValueError, match="unknown backend mode"):
        select_final_output("bogus", combinator, protocol_b)


def test_shadow_comparison_records_both_sides_and_diff_summary():
    from src.pipeline.protocol_b_adapter import build_shadow_comparison

    combinator, protocol_b = _results()

    summary = build_shadow_comparison(
        combinator_result=combinator,
        protocol_b_result=protocol_b,
        combinator_elapsed_ms=12.0,
        protocol_b_elapsed_ms=34.0,
    )

    assert summary["combinator"]["models"] == ["m1"]
    assert summary["protocol_b"]["models"] == ["m1", "m2"]
    assert summary["combinator"]["weights"] == {"m1": 1.0}
    assert summary["protocol_b"]["weights"] == {"m1": 0.5, "m2": 0.5}
    assert summary["combinator"]["elapsed_ms"] == 12.0
    assert summary["protocol_b"]["elapsed_ms"] == 34.0
    assert summary["diff"]["selection_changed"] is True
    assert summary["diff"]["mae_delta"] == pytest.approx(0.4 - 0.5)
    assert summary["final_output_from"] == "combinator"


def test_shadow_comparison_marks_identical_selection():
    from src.pipeline.protocol_b_adapter import build_shadow_comparison

    combinator, _ = _results()

    summary = build_shadow_comparison(
        combinator_result=combinator,
        protocol_b_result=dict(combinator),
        combinator_elapsed_ms=1.0,
        protocol_b_elapsed_ms=2.0,
    )

    assert summary["diff"]["selection_changed"] is False
    assert summary["diff"]["mae_delta"] == pytest.approx(0.0)


# --- Task 3.1: adapter 必须使用引擎实际预测 ----------------------------------


def test_adapter_uses_engine_predictions_not_linear_reconstruction(monkeypatch, tmp_path):
    """引擎交出预测时，yhat 必须来自引擎，且与线性重建明显不同。"""
    from src.eval.kg.config import RUNTIME_PREDICTIONS_KEY

    bundle = _bundle()
    raw = _fake_raw(bundle, selected=["m1"], weights={"m1": 1.0})
    engine_pred = bundle.df_test["m1"].values + 3.0  # 故意偏离线性重建
    y = bundle.df_test["y"].values
    raw["test"]["mae"] = float(np.mean(np.abs(engine_pred - y)))
    raw[RUNTIME_PREDICTIONS_KEY] = {
        "val": bundle.df_val["m1"].values,
        "test": engine_pred,
    }
    _patch_protocol_b(monkeypatch, raw)

    result = DemoProtocolBAdapter().select(bundle, region=REGION, horizon=1)

    assert result["yhat_source"] == "engine"
    np.testing.assert_allclose(result["yhat"], engine_pred, atol=1e-12)
    assert result["mae"] == pytest.approx(raw["test"]["mae"], abs=1e-9)
    assert result["mae_matches_protocol_b"] is True
    # 线性重建对不上，但只作为诊断记录，不覆盖真实预测
    assert result["linear_reconstruction_match"] is False
    assert result["linear_reconstruction_mae"] != pytest.approx(result["mae"])


def test_runtime_predictions_never_reach_trace_or_raw(monkeypatch, tmp_path):
    """运行时预测不得写入 SelectionTrace 或留在 raw 里（否则会进实验 JSON）。"""
    import json

    from src.eval.kg.config import RUNTIME_PREDICTIONS_KEY

    bundle = _bundle()
    raw = _fake_raw(bundle, selected=["m1"], weights={"m1": 1.0})
    raw[RUNTIME_PREDICTIONS_KEY] = {
        "val": bundle.df_val["m1"].values,
        "test": bundle.df_test["m1"].values,
    }
    _patch_protocol_b(monkeypatch, raw)

    result = DemoProtocolBAdapter().select(
        bundle, region=REGION, horizon=1, trace_path=tmp_path / "t.json"
    )

    assert RUNTIME_PREDICTIONS_KEY not in (result["raw"] or {})
    saved = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert RUNTIME_PREDICTIONS_KEY not in json.dumps(saved)
    # trace 里只留核对摘要，不留数组
    stage = _find_stage(result["trace"], "DemoProtocolBAdapter")
    for value in stage["outputs"].values():
        assert not isinstance(value, np.ndarray)


def test_real_engine_interaction_branch_adapter_uses_actual_predictions(monkeypatch, tmp_path):
    """端到端：真实引擎强制走"交互已应用、post_adjustment 未应用"分支。

    该分支下线性重建与引擎上报 MAE 必然不同；断言 adapter 输出的 yhat 对应的是
    引擎真实预测（其 MAE 精确等于上报值），而不是线性重建值。
    """
    import src.eval.kg.protocol_b as pb
    from src.pipeline.prediction_pool import RegionPredictionBundle

    monkeypatch.setattr(pb, "PROTOCOL_B_ADJUST_BONUS_SCALE", 50.0)

    n_val, n_test = 1500, 300
    tsv = pd.date_range("2026-01-01", periods=n_val, freq="h")
    tst = pd.date_range("2026-06-01", periods=n_test, freq="h")

    def mk(ts, n, seed):
        r = np.random.default_rng(seed)
        temp = np.linspace(5, 35, n) + r.normal(0, 1, n)
        y = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + 1.5 * temp + r.normal(0, 1, n)
        df = pd.DataFrame({
            "timestamp": ts, "y": y,
            "m1": y - 0.9 * temp + r.normal(0, 2, n),
            "m2": y + 0.7 * temp + r.normal(0, 3, n),
            "m3": y + r.normal(0, 6, n),
        })
        rawf = pd.DataFrame({"timestamp": ts, "hour": ts.hour, "dow": ts.dayofweek, "temp": temp})
        return df, rawf

    dv, rv = mk(tsv, n_val, 1)
    dt, rt = mk(tst, n_test, 2)
    bundle = RegionPredictionBundle(
        df_val=dv, df_test=dt, df_raw_val=rv, df_raw_test=rt,
        model_cols=["m1", "m2", "m3"], base_model_cols=["m1"],
        fitted_test_models={}, metadata={},
    )

    result = DemoProtocolBAdapter().select(bundle, region="T31", horizon=1)

    assert result["strategy"] == "B_pred_features"
    assert result["yhat_source"] == "engine"
    # 该分支的线性重建确实对不上（这正是 Task 3.1 存在的理由）
    assert result["linear_reconstruction_match"] is False
    # 而 adapter 报的 MAE 由实际 yhat 算出，且精确等于引擎上报值
    assert result["mae"] == pytest.approx(result["protocol_b_mae"], abs=1e-9)
    assert result["mae"] == pytest.approx(
        float(np.mean(np.abs(result["yhat"] - dt["y"].values))), abs=1e-12
    )
    assert abs(result["linear_reconstruction_mae"] - result["mae"]) > 1e-8
