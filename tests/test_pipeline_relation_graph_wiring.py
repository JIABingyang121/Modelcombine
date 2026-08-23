"""Pipeline 层：同一张图必须传到 Adapter，反馈必须用真实 scenario_id。

Adapter 全链路测试（`test_adapter_relation_strength_causal.py`）证明不了这一层：
`main.py` 完全可能不传图、或继续用写死的 `protocol_b::{region}` 作反馈场景 ID，
而 Adapter 侧测试照样全绿。故本模块断言打在 `PowerPredictionPipeline` 上。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.pipeline.main as pipeline_main
from src.pipeline.main import PowerPredictionPipeline

REGION = "R1"


def _frame(n: int = 240):
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": ts, "region": REGION, "region_type": "residential",
        "load": np.linspace(100.0, 200.0, n) + rng.normal(0, 1.0, n),
        "hour": ts.hour, "dow": ts.dayofweek,
        "is_weekend": ts.dayofweek.isin([5, 6]).astype(int),
        "lag_1": np.linspace(99.0, 199.0, n),
    })


@pytest.fixture()
def wired(monkeypatch, tmp_path):
    """构造只保留区域循环的流水线，记录传给 adapter 的图与反馈场景 ID。"""
    captured = {}

    pipeline = PowerPredictionPipeline.__new__(PowerPredictionPipeline)
    pipeline.config = {
        "pipeline": {
            "data": {"regions": [REGION], "test_days": 2, "validation_days": 2,
                     "root": "data/demo"},
            "models": {}, "selection": {"combiner": "weighted"},
        },
        "assets": {"models": [], "relations": [], "selection_rules": []},
    }
    pipeline.enable_phase2 = False
    pipeline.enable_phase3 = False
    pipeline.historical_scenarios = []

    df = _frame()
    train, test = df.iloc[:192], df.iloc[192:]

    class _Graph:
        def save_graph(self, path):
            pass

    graph = _Graph()
    monkeypatch.setattr(pipeline, "ensure_power_data", lambda: None, raising=False)
    monkeypatch.setattr(pipeline, "build_power_features", lambda: df, raising=False)
    monkeypatch.setattr(pipeline, "split_train_test",
                        lambda d, test_days: (train, test), raising=False)
    monkeypatch.setattr(pipeline, "build_model_graph", lambda: graph, raising=False)
    monkeypatch.setattr(pipeline, "evaluate_results",
                        lambda p, q: {"overall": {"RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0},
                                      "by_region": {}}, raising=False)

    def fake_feedback(evaluation, model_graph, sid_map, pid_map, perfs):
        captured["feedback_scenario_ids"] = dict(sid_map or {})
        captured["feedback_graph"] = model_graph

    monkeypatch.setattr(pipeline, "feedback_loop", fake_feedback, raising=False)
    monkeypatch.setattr(pipeline, "_save_phase2_models", lambda: None, raising=False)
    monkeypatch.setattr(pipeline, "_save_phase3_models", lambda: None, raising=False)
    monkeypatch.setattr(pipeline_main, "PROJECT_ROOT", str(tmp_path))

    from src.pipeline.prediction_pool import RegionPredictionBundle

    def fake_bundle(**kwargs):
        te = kwargs["test"]
        return RegionPredictionBundle(
            df_val=pd.DataFrame({"timestamp": train["timestamp"].values[-24:],
                                 "y": train["load"].values[-24:],
                                 "m1": train["load"].values[-24:]}),
            df_test=pd.DataFrame({"timestamp": te["timestamp"].values,
                                  "y": te["load"].values, "m1": te["load"].values}),
            df_raw_val=pd.DataFrame({"timestamp": train["timestamp"].values[-24:]}),
            df_raw_test=pd.DataFrame({"timestamp": te["timestamp"].values}),
            model_cols=["m1"], base_model_cols=["m1"],
            fitted_test_models={}, metadata={},
        )

    monkeypatch.setattr(pipeline_main, "build_region_prediction_bundle",
                        fake_bundle, raising=False)

    class _FakeAdapter:
        def select(self, bundle, *, region, horizon=1, trace_path=None, model_graph=None):
            captured["adapter_model_graph"] = model_graph
            return {
                "models": ["m1"], "weights": {"m1": 1.0}, "strategy": "B_pred_features",
                "path_id": "B_pred_features",
                "yhat": np.linspace(1.0, 2.0, len(bundle.df_test)),
                "yhat_source": "engine", "trace": None, "mae": 1.0,
                "protocol_b_mae": 1.0, "mae_delta": 0.0,
                "mae_matches_protocol_b": True, "linear_reconstruction_mae": 1.0,
                "linear_reconstruction_match": True, "reconcile_note": None,
                "feedback_store": None,
                "scenario_id": "real_scenario_hash_abc123",
                "raw": {},
            }

    monkeypatch.setattr(pipeline_main, "DemoProtocolBAdapter", _FakeAdapter, raising=False)
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_BACKEND", raising=False)
    return pipeline, graph, captured


def test_pipeline_passes_the_same_graph_to_adapter(wired):
    """main.py 必须把 build_model_graph() 的同一张图传给 adapter。"""
    pipeline, graph, captured = wired

    pipeline.run_prediction_pipeline()

    assert captured["adapter_model_graph"] is graph, (
        "adapter 收到的不是主流程那张图——关系强度在默认 run.py 路径上不会生效"
    )


def test_feedback_uses_real_scenario_id_not_placeholder(wired):
    """反馈必须用 adapter 返回的真实 scenario_id，而不是 protocol_b::{region}。"""
    pipeline, _graph, captured = wired

    pipeline.run_prediction_pipeline()

    sid = captured["feedback_scenario_ids"][REGION]
    assert sid == "real_scenario_hash_abc123", f"反馈场景 ID 不是真实值：{sid}"
    assert not str(sid).startswith("protocol_b::"), (
        "仍在使用写死的占位场景 ID，反馈边会落在没人读的场景上"
    )


def test_feedback_receives_the_same_graph(wired):
    """反馈阶段与消费阶段必须操作同一张图，否则写入的边下一轮读不到。"""
    pipeline, graph, captured = wired

    pipeline.run_prediction_pipeline()

    assert captured["feedback_graph"] is graph
