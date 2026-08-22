"""三态后端在 run.py 主流程中的数据流（System A/B 合一 Task 4）。

本模块先用**特征化测试**锁定改造前 combinator 默认流程的可观测行为（调用顺序、
yhat、反馈次数、产物），再验证新增的 Protocol B 独立流程与影子模式：

- `combinator`（默认）：行为必须与改造前逐项一致；
- `protocol_b_shadow`：两条路都跑，最终输出仍取 combinator 且逐值一致，
  Protocol B 侧只产审计，不得写任何生产状态；
- `protocol_b`：输出取 Protocol B，且必须来自运行时精确预测
  （`yhat_source == "engine"`），禁止退回权重线性重建。

实际输出路径只反馈一次，不允许 combinator 与 Protocol B 双写。
"""
import json

import numpy as np
import pandas as pd
import pytest

import src.pipeline.main as pipeline_main
from src.pipeline.main import PowerPredictionPipeline

REGION = "R1"


# --- 测试脚手架 --------------------------------------------------------------


def _frame(n=240, start="2026-01-01"):
    ts = pd.date_range(start, periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "region": REGION,
            "region_type": "residential",
            "load": np.linspace(100.0, 200.0, n) + rng.normal(0, 1.0, n),
            "hour": ts.hour,
            "dow": ts.dayofweek,
            "is_weekend": ts.dayofweek.isin([5, 6]).astype(int),
            "day": ts.day,
            "month": ts.month,
            "is_holiday": np.zeros(n, dtype=int),
            "lag_1": np.linspace(99.0, 199.0, n),
        }
    )


class _Recorder:
    """记录主流程关键调用的顺序与次数。"""

    def __init__(self):
        self.calls = []

    def log(self, name, **kw):
        self.calls.append((name, kw))

    def names(self):
        return [c[0] for c in self.calls]

    def count(self, name):
        return sum(1 for c in self.calls if c[0] == name)


def _install_pipeline(monkeypatch, tmp_path, recorder, *, protocol_b_yhat=None):
    """构造一个只保留区域循环逻辑的流水线，其余重活全部替身。"""
    pipeline = PowerPredictionPipeline.__new__(PowerPredictionPipeline)
    pipeline.config = {
        "pipeline": {
            "data": {"regions": [REGION], "test_days": 2, "validation_days": 2, "root": "data/demo"},
            "models": {},
            "selection": {"combiner": "weighted"},
        },
        "assets": {"models": [], "relations": [], "selection_rules": []},
    }
    pipeline.enable_phase2 = False
    pipeline.enable_phase3 = False
    pipeline.historical_scenarios = []

    df = _frame()
    train, test = df.iloc[:192], df.iloc[192:]

    monkeypatch.setattr(pipeline, "ensure_power_data", lambda: None, raising=False)
    monkeypatch.setattr(pipeline, "load_data", lambda: df, raising=False)
    monkeypatch.setattr(pipeline, "build_power_features", lambda: df, raising=False)
    monkeypatch.setattr(
        pipeline, "split_train_test", lambda d, test_days: (train, test), raising=False
    )

    class _Graph:
        def __init__(self):
            self.mutations = []

        def save_graph(self, path):
            recorder.log("graph.save", path=str(path))

    graph = _Graph()
    monkeypatch.setattr(pipeline, "build_model_graph", lambda: graph, raising=False)

    def fake_select(region, df_train, model_graph):
        recorder.log("select_models_for_region", region=region)
        return ["m1", "m2"], {"m1": 0.6, "m2": 0.4}, "sid_1", "path_1"

    monkeypatch.setattr(pipeline, "select_models_for_region", fake_select, raising=False)

    combinator_yhat = np.linspace(150.0, 160.0, len(test))

    def fake_fit_predict(region, selected_models, weights, tr, te):
        recorder.log("fit_and_predict_region", region=region, models=list(selected_models))
        out = te[["timestamp", "region", "load"]].copy()
        out["yhat"] = combinator_yhat
        return out, {"m1": {"RMSE": 1.0}, "_profiling": {"elapsed_seconds": 0.1}}

    monkeypatch.setattr(pipeline, "fit_and_predict_region", fake_fit_predict, raising=False)

    def fake_evaluate(pred_df, perfs):
        recorder.log("evaluate_results")
        return {"overall": {"RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0}, "by_region": {}}

    monkeypatch.setattr(pipeline, "evaluate_results", fake_evaluate, raising=False)

    def fake_feedback(evaluation, model_graph, sid_map, pid_map, perfs):
        recorder.log("feedback_loop", scenario_ids=dict(sid_map or {}))

    monkeypatch.setattr(pipeline, "feedback_loop", fake_feedback, raising=False)
    monkeypatch.setattr(pipeline, "_save_phase2_models", lambda: None, raising=False)
    monkeypatch.setattr(pipeline, "_save_phase3_models", lambda: None, raising=False)
    monkeypatch.setattr(pipeline_main, "PROJECT_ROOT", str(tmp_path))

    # Protocol B 侧替身：bundle 构建与 adapter
    def fake_bundle(**kwargs):
        recorder.log(
            "build_region_prediction_bundle",
            region=kwargs.get("region"),
            candidate_models=list(kwargs.get("candidate_models") or []),
        )
        from src.pipeline.prediction_pool import RegionPredictionBundle

        te = kwargs["test"]
        return RegionPredictionBundle(
            df_val=pd.DataFrame(
                {"timestamp": train["timestamp"].values[-24:], "y": train["load"].values[-24:],
                 "m1": train["load"].values[-24:], "m2": train["load"].values[-24:]}
            ),
            df_test=pd.DataFrame(
                {"timestamp": te["timestamp"].values, "y": te["load"].values,
                 "m1": te["load"].values, "m2": te["load"].values}
            ),
            df_raw_val=pd.DataFrame({"timestamp": train["timestamp"].values[-24:]}),
            df_raw_test=pd.DataFrame({"timestamp": te["timestamp"].values}),
            model_cols=["m1", "m2"],
            base_model_cols=["m1"],
            fitted_test_models={},
            metadata={},
        )

    monkeypatch.setattr(pipeline_main, "build_region_prediction_bundle", fake_bundle, raising=False)

    pb_yhat = protocol_b_yhat if protocol_b_yhat is not None else np.linspace(900.0, 910.0, len(test))

    class _FakeAdapter:
        def select(self, bundle, *, region, horizon=1, trace_path=None):
            recorder.log("protocol_b_adapter.select", region=region)
            return {
                "models": ["m1"],
                "weights": {"m1": 1.0},
                "strategy": "B_pred_features",
                "path_id": "B_pred_features",
                "yhat": pb_yhat,
                "yhat_source": "engine",
                "trace": None,
                "mae": 2.0,
                "protocol_b_mae": 2.0,
                "mae_delta": 0.0,
                "mae_matches_protocol_b": True,
                "linear_reconstruction_mae": 9.0,
                "linear_reconstruction_match": False,
                "reconcile_note": None,
                "feedback_store": object(),
                "raw": {},
            }

    monkeypatch.setattr(pipeline_main, "DemoProtocolBAdapter", _FakeAdapter, raising=False)

    return pipeline, train, test, combinator_yhat, pb_yhat


def _read_predictions(tmp_path):
    return pd.read_csv(tmp_path / "reports" / "predictions.csv")


# --- 特征化测试：改造前 combinator 默认流程 ---------------------------------


def test_characterize_combinator_default_call_order(monkeypatch, tmp_path):
    """锁定默认流程的调用顺序：选模型 -> 训练预测 -> 评估 -> 反馈 -> 存图谱。"""
    # Task 8 起 combinator 不再是默认，必须显式指定才走旧路径
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "combinator")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    assert recorder.names() == [
        "select_models_for_region",
        "fit_and_predict_region",
        "evaluate_results",
        "feedback_loop",
        "graph.save",
    ]


def test_characterize_combinator_default_outputs(monkeypatch, tmp_path):
    """锁定默认流程的产物：predictions.csv 的 yhat、report.json、model_info.json。"""
    # Task 8 起 combinator 不再是默认，必须显式指定才走旧路径
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "combinator")
    recorder = _Recorder()
    pipeline, _, test, combinator_yhat, _ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    preds = _read_predictions(tmp_path)
    assert list(preds.columns) == ["timestamp", "region", "load", "yhat"]
    np.testing.assert_allclose(preds["yhat"].values, combinator_yhat, atol=1e-12)
    assert (tmp_path / "reports" / "report.json").exists()
    assert (tmp_path / "reports" / "model_info.json").exists()


def test_characterize_combinator_feedback_written_exactly_once(monkeypatch, tmp_path):
    # Task 8 起 combinator 不再是默认，必须显式指定才走旧路径
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "combinator")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    assert recorder.count("feedback_loop") == 1
    assert recorder.count("fit_and_predict_region") == 1
    # 默认路径完全不碰 Protocol B
    assert recorder.count("protocol_b_adapter.select") == 0
    assert recorder.count("build_region_prediction_bundle") == 0


# --- protocol_b 模式 ---------------------------------------------------------


def test_protocol_b_mode_call_order_predicts_before_combining(monkeypatch, tmp_path):
    """Protocol B 模式：先构候选预测矩阵，再调 solver；不得走旧 combinator 决策。"""
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    names = recorder.names()
    assert names.index("build_region_prediction_bundle") < names.index("protocol_b_adapter.select")
    assert recorder.count("fit_and_predict_region") == 0
    assert recorder.count("feedback_loop") == 1


def test_protocol_b_mode_output_comes_from_engine_predictions(monkeypatch, tmp_path):
    """最终 yhat 必须来自运行时精确预测，禁止退回线性重建。"""
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b")
    recorder = _Recorder()
    pipeline, _, _, combinator_yhat, pb_yhat = _install_pipeline(monkeypatch, tmp_path, recorder)

    result = pipeline.run_prediction_pipeline()

    preds = _read_predictions(tmp_path)
    np.testing.assert_allclose(preds["yhat"].values, pb_yhat, atol=1e-12)
    assert not np.allclose(preds["yhat"].values, combinator_yhat)
    assert result["backend"]["mode"] == "protocol_b"
    assert result["backend"]["regions"][REGION]["yhat_source"] == "engine"


def test_protocol_b_mode_rejects_linear_reconstruction_fallback(monkeypatch, tmp_path):
    """adapter 若退回线性重建，主流程必须报错而不是照单全收。"""
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    class _LinearAdapter:
        def select(self, bundle, *, region, horizon=1, trace_path=None):
            return {
                "models": ["m1"], "weights": {"m1": 1.0}, "strategy": "B_pred_features",
                "path_id": "B_pred_features", "yhat": np.zeros(len(bundle.df_test)),
                "yhat_source": "linear_reconstruction", "trace": None, "mae": 1.0,
                "protocol_b_mae": 1.0, "mae_delta": 0.0, "mae_matches_protocol_b": True,
                "linear_reconstruction_mae": 1.0, "linear_reconstruction_match": True,
                "reconcile_note": None, "feedback_store": None, "raw": {},
            }

    monkeypatch.setattr(pipeline_main, "DemoProtocolBAdapter", _LinearAdapter, raising=False)

    with pytest.raises(RuntimeError, match="yhat_source"):
        pipeline.run_prediction_pipeline()


def test_protocol_b_failure_raises_and_does_not_fall_back(monkeypatch, tmp_path):
    """Protocol B 失败必须显式报错，不得静默切回 combinator。"""
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    class _BoomAdapter:
        def select(self, *a, **k):
            raise RuntimeError("protocol b exploded")

    monkeypatch.setattr(pipeline_main, "DemoProtocolBAdapter", _BoomAdapter, raising=False)

    with pytest.raises(RuntimeError, match="protocol b exploded"):
        pipeline.run_prediction_pipeline()

    assert recorder.count("fit_and_predict_region") == 0
    assert recorder.count("feedback_loop") == 0


# --- protocol_b_shadow 模式 --------------------------------------------------


def test_shadow_mode_runs_both_but_outputs_combinator(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    recorder = _Recorder()
    pipeline, _, _, combinator_yhat, pb_yhat = _install_pipeline(monkeypatch, tmp_path, recorder)

    result = pipeline.run_prediction_pipeline()

    assert recorder.count("fit_and_predict_region") == 1
    assert recorder.count("protocol_b_adapter.select") == 1
    preds = _read_predictions(tmp_path)
    # 逐值一致于 combinator，且明显不同于 Protocol B
    np.testing.assert_allclose(preds["yhat"].values, combinator_yhat, atol=1e-12)
    assert not np.allclose(preds["yhat"].values, pb_yhat)
    assert result["backend"]["mode"] == "protocol_b_shadow"
    assert result["backend"]["regions"][REGION]["final_output_from"] == "combinator"


def test_shadow_mode_output_identical_to_combinator_mode(monkeypatch, tmp_path):
    """影子模式的 predictions.csv 必须与默认模式逐值一致。"""
    recorder_a = _Recorder()
    # 基线取显式 combinator（Task 8 后它不再是默认值）
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "combinator")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    p1, *_ = _install_pipeline(monkeypatch, base_dir, recorder_a)
    p1.run_prediction_pipeline()
    baseline = _read_predictions(base_dir)

    recorder_b = _Recorder()
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    p2, *_ = _install_pipeline(monkeypatch, shadow_dir, recorder_b)
    p2.run_prediction_pipeline()
    shadow = _read_predictions(shadow_dir)

    pd.testing.assert_frame_equal(baseline, shadow)


def test_shadow_mode_feedback_written_exactly_once(monkeypatch, tmp_path):
    """只有实际输出路径可写反馈，不能 combinator 与 Protocol B 双写。"""
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    assert recorder.count("feedback_loop") == 1


def test_shadow_mode_does_not_touch_production_state(monkeypatch, tmp_path):
    """影子路径不得改动历史场景库、生产图谱、旧权重/资源模型与生产反馈存储。"""
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    hist_before = json.dumps(pipeline.historical_scenarios, sort_keys=True, default=str)

    class _Tripwire:
        def __init__(self):
            self.touched = []

        def add_observation(self, *a, **k):
            self.touched.append("add_observation")

        def fit(self, *a, **k):
            self.touched.append("fit")

    tripwire = _Tripwire()

    class _Combinator:
        resource_predictor = tripwire
        weight_manager = tripwire

    pipeline.model_combinator = _Combinator()

    pipeline.run_prediction_pipeline()

    assert json.dumps(pipeline.historical_scenarios, sort_keys=True, default=str) == hist_before
    # Protocol B 侧未通过旧组合器写任何学习状态
    assert tripwire.touched == []
    # 图谱只由主路径在收尾时保存一次
    assert recorder.count("graph.save") == 1


def test_shadow_mode_records_comparison_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    result = pipeline.run_prediction_pipeline()

    summary = result["backend"]["regions"][REGION]["comparison"]
    assert summary["final_output_from"] == "combinator"
    assert summary["protocol_b"]["models"] == ["m1"]
    assert "mae_delta" in summary["diff"]


def test_shadow_mode_protocol_b_failure_still_raises(monkeypatch, tmp_path):
    """影子模式下 Protocol B 失败同样必须暴露，不能因为"只是审计"就吞掉。"""
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    class _BoomAdapter:
        def select(self, *a, **k):
            raise RuntimeError("shadow protocol b exploded")

    monkeypatch.setattr(pipeline_main, "DemoProtocolBAdapter", _BoomAdapter, raising=False)

    with pytest.raises(RuntimeError, match="shadow protocol b exploded"):
        pipeline.run_prediction_pipeline()


def test_combinator_mode_produces_no_new_artifacts(monkeypatch, tmp_path):
    """默认路径产物集合必须与改造前一致：不得多出 backend_report.json。"""
    # Task 8 起 combinator 不再是默认，必须显式指定才走旧路径
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "combinator")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    files = sorted(p.name for p in (tmp_path / "reports").iterdir() if p.is_file())
    assert files == ["model_info.json", "predictions.csv", "report.json"]


def test_combinator_report_json_not_polluted_by_backend_key(monkeypatch, tmp_path):
    """report.json 在第 9 步保存，backend 字段只应存在于返回值中。"""
    # Task 8 起 combinator 不再是默认，必须显式指定才走旧路径
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "combinator")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    saved = json.loads((tmp_path / "reports" / "report.json").read_text(encoding="utf-8"))
    assert "backend" not in saved


def test_non_default_modes_write_backend_report(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    assert (tmp_path / "reports" / "backend_report.json").exists()


# --- Task 8: 默认决策路径切换到 Protocol B ---------------------------------


def test_default_backend_is_protocol_b_without_env(monkeypatch):
    """未设置环境变量时，默认必须是 protocol_b（Task 8 切换的核心断言）。"""
    from src.pipeline.main import resolve_backend_mode

    monkeypatch.delenv("MODELCOMBINE_PIPELINE_BACKEND", raising=False)

    assert resolve_backend_mode() == "protocol_b"


def test_empty_or_whitespace_env_also_defaults_to_protocol_b(monkeypatch):
    """空串/空白不得被当成"未设置以外的东西"而落回旧引擎。"""
    from src.pipeline.main import resolve_backend_mode

    for raw in ("", "   "):
        monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", raw)
        assert resolve_backend_mode() == "protocol_b"


def test_legacy_combinator_requires_explicit_opt_in(monkeypatch):
    """只有显式设置 combinator 才走旧路径；旧路径不再是隐式默认。"""
    from src.pipeline.main import resolve_backend_mode

    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "combinator")
    assert resolve_backend_mode() == "combinator"

    monkeypatch.setenv("MODELCOMBINE_PIPELINE_BACKEND", "protocol_b_shadow")
    assert resolve_backend_mode() == "protocol_b_shadow"


def test_default_run_uses_protocol_b_path_end_to_end(monkeypatch, tmp_path):
    """默认（无环境变量）跑主流程时，必须真正走 Protocol B 独立流程。"""
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_BACKEND", raising=False)
    recorder = _Recorder()
    pipeline, _, _, combinator_yhat, pb_yhat = _install_pipeline(monkeypatch, tmp_path, recorder)

    result = pipeline.run_prediction_pipeline()

    assert recorder.count("protocol_b_adapter.select") == 1
    assert recorder.count("fit_and_predict_region") == 0
    preds = _read_predictions(tmp_path)
    np.testing.assert_allclose(preds["yhat"].values, pb_yhat, atol=1e-12)
    assert result["backend"]["mode"] == "protocol_b"
    assert result["backend"]["regions"][REGION]["yhat_source"] == "engine"
    # 反馈仍只写一次
    assert recorder.count("feedback_loop") == 1


def test_default_run_passes_only_release_approved_candidates_to_protocol_b(monkeypatch, tmp_path):
    """Task 8.1：默认路径不得把未完成服务器验收的深度候选交给预测池。"""
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_BACKEND", raising=False)
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    bundle_call = next(details for name, details in recorder.calls if name == "build_region_prediction_bundle")
    assert {"informer", "autoformer", "powergpt"}.isdisjoint(bundle_call["candidate_models"])


def test_default_run_raises_instead_of_silently_using_combinator(monkeypatch, tmp_path):
    """默认路径下 Protocol B 失败必须报错，禁止静默回退到旧引擎。"""
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_BACKEND", raising=False)
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    class _BoomAdapter:
        def select(self, *a, **k):
            raise RuntimeError("protocol b exploded under new default")

    monkeypatch.setattr(pipeline_main, "DemoProtocolBAdapter", _BoomAdapter, raising=False)

    with pytest.raises(RuntimeError, match="protocol b exploded under new default"):
        pipeline.run_prediction_pipeline()

    assert recorder.count("fit_and_predict_region") == 0
    assert recorder.count("feedback_loop") == 0


def test_model_info_records_actual_backend_and_fallback_state(monkeypatch, tmp_path):
    """model_info.json 必须记录实际后端、是否影子、是否回退与 trace 路径。"""
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_BACKEND", raising=False)
    recorder = _Recorder()
    pipeline, *_ = _install_pipeline(monkeypatch, tmp_path, recorder)

    pipeline.run_prediction_pipeline()

    info = json.loads((tmp_path / "reports" / "model_info.json").read_text(encoding="utf-8"))
    backend = info.get("backend")
    assert backend is not None, "model_info.json 必须记录 backend 信息"
    assert backend["mode"] == "protocol_b"
    assert backend["is_shadow"] is False
    assert backend["fell_back_to_combinator"] is False
    assert REGION in backend["regions"]
    assert "trace_path" in backend["regions"][REGION]
