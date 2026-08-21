import json

import numpy as np
import pandas as pd

import scripts.train_combinations_kg as runner
from src.core.trace import SelectionTrace
from src.graph.model_graph import ModelGraph


def _frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "y": np.asarray([10.0, 11.0, 12.0, 13.0]),
            "m1": np.asarray([10.1, 11.1, 12.1, 13.1]),
            "m2": np.asarray([9.9, 10.9, 11.9, 12.9]),
        }
    )


def test_protocol_b_context_preserves_safe_model_pool():
    # 上下文构造已抽取到 src/core/solver/protocol_b_context.py（System A/B 合一 Task 1）；
    # 这里经 runner 命名空间调用，确认脚本确实走的是那份共用实现而非自带副本。
    ctx = runner.build_protocol_b_context(
        dataset="pjm",
        horizon=1,
        df_val=_frame(),
        df_test=_frame(),
        df_raw_val=pd.DataFrame({"load": [1.0, 2.0]}),
        df_raw_test=pd.DataFrame({"load": [3.0, 4.0]}),
        model_cols=["m1", "m2"],
        base_model_cols=["m1"],
        feedback_store=None,
    )

    assert ctx.scenario.business_domain == "load_forecast"
    assert ctx.scenario.primary_metric == "MAE"
    assert ctx.model_cols == ["m1", "m2"]
    assert ctx.base_model_cols == ["m1"]
    assert "load" in ctx.available_features


def test_run_protocol_b_with_solver_saves_trace_and_unwraps_raw(monkeypatch, tmp_path):
    raw = {
        "val": {"mae": 1.0, "selected_models": ["m1"], "weights": {"m1": 1.0}},
        "test": {"mae": 1.2, "selected_models": ["m1"], "weights": {"m1": 1.0}},
        "protocol": "kg_protocol_b",
    }

    class FakeSolver:
        def solve(self, ctx, trace_path=None):
            assert ctx.model_cols == ["m1", "m2"]
            trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)
            trace.add_stage("CapabilityMatch")
            trace.add_stage("SimilarScenarioRetrieve")
            trace.add_stage("ProtocolBBackend")
            trace.add_stage("UncertaintyEstimate")
            if trace_path:
                trace.save_json(trace_path)
            return {"raw": raw, "models": ["m1"], "weights": {"m1": 1.0}}, trace

    def fake_build_solver(mode, *, manifests, index_manager, uncertainty_gate):
        assert mode == "protocol_b"
        assert sorted(manifests) == ["m1", "m2"]
        assert "scenario" in index_manager.names()
        if index_manager.query("latency", scenario_id="pjm_h1"):
            assert index_manager.query("drift", scenario_id="pjm_h1")
        assert uncertainty_gate is not None
        return FakeSolver()

    monkeypatch.setattr(runner, "build_solver", fake_build_solver)
    trace_path = tmp_path / "trace.json"
    kg_results = tmp_path / "kg_results.json"
    kg_results.write_text(
        json.dumps(
            {
                "pjm": {
                    "1": {
                        "_meta": {
                            "runtime_protocol_b_sec": 0.1,
                            "protocol_b": {
                                "guard_config": {
                                    "drift_level_effective": "low",
                                    "drift_median_psi": 0.01,
                                }
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result, trace = runner._run_protocol_b_with_solver(
        dataset="pjm",
        horizon=1,
        df_val=_frame(),
        df_test=_frame(),
        df_raw_val=None,
        df_raw_test=None,
        model_cols=["m1", "m2"],
        base_model_cols=["m1", "m2"],
        feedback_store=None,
        trace_path=trace_path,
        signal_kg_result_paths=[kg_results],
    )

    assert result is raw
    assert [s["stage"] for s in trace.stages] == [
        "CapabilityMatch",
        "SimilarScenarioRetrieve",
        "ProtocolBBackend",
        "UncertaintyEstimate",
    ]
    saved = json.loads(trace_path.read_text(encoding="utf-8"))
    assert saved["stages"][0]["stage"] == "CapabilityMatch"


def _capture_build_solver(captured):
    def fake_build_solver(mode, **kwargs):
        captured["mode"] = mode
        captured["kwargs"] = kwargs

        class FakeSolver:
            def solve(self, ctx, trace_path=None):
                trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)
                trace.add_stage("ProtocolBBackend")
                return {"raw": {}, "models": [], "weights": {}}, trace

        return FakeSolver()

    return fake_build_solver


def test_temporal_relation_graph_not_wired_by_default(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(runner, "build_solver", _capture_build_solver(captured))

    runner._run_protocol_b_with_solver(
        dataset="pjm",
        horizon=1,
        df_val=_frame(),
        df_test=_frame(),
        df_raw_val=None,
        df_raw_test=None,
        model_cols=["m1", "m2"],
        base_model_cols=["m1"],
        feedback_store=None,
    )

    assert captured["mode"] == "protocol_b"
    assert "temporal_relation_graph" not in captured["kwargs"]


def test_temporal_relation_graph_wired_when_provided(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(runner, "build_solver", _capture_build_solver(captured))
    graph = ModelGraph()

    runner._run_protocol_b_with_solver(
        dataset="pjm",
        horizon=1,
        df_val=_frame(),
        df_test=_frame(),
        df_raw_val=None,
        df_raw_test=None,
        model_cols=["m1", "m2"],
        base_model_cols=["m1"],
        feedback_store=None,
        temporal_relation_graph=graph,
    )

    kwargs = captured["kwargs"]
    assert kwargs["temporal_relation_graph"] is graph
    assert kwargs["temporal_relation_create_missing"] is True
    assert kwargs["temporal_relation_updater"] is not None


def test_temporal_relations_switch_reads_env(monkeypatch):
    monkeypatch.delenv("MODELCOMBINE_KG_ENABLE_TEMPORAL_RELATIONS", raising=False)
    assert runner._temporal_relations_enabled() is False
    monkeypatch.setenv("MODELCOMBINE_KG_ENABLE_TEMPORAL_RELATIONS", "true")
    assert runner._temporal_relations_enabled() is True


def test_temporal_relation_updater_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("MODELCOMBINE_KG_TEMPORAL_ALPHA", "0.35")
    monkeypatch.setenv("MODELCOMBINE_KG_TEMPORAL_BASE_STRENGTH", "0.4")
    updater = runner._build_temporal_relation_updater()
    assert updater.alpha == 0.35
    assert updater.base_strength == 0.4
