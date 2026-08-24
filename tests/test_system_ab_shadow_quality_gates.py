"""Task 7 修正：同矩阵 System A 参考与机器可执行质量门槛。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_system_ab_shadow as shadow
from src.pipeline.main import PowerPredictionPipeline


def test_system_a_reference_blends_the_shared_horizon_matrix_without_renormalizing():
    """防止 System A 又回到独立训练，或悄悄改变旧权重语义。"""
    df_test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h"),
            "y": [10.0, 20.0],
            "m1": [12.0, 18.0],
            "m2": [8.0, 22.0],
        }
    )

    result = shadow.evaluate_system_a_matrix_reference(
        dataset="pjm",
        horizon=6,
        df_test=df_test,
        selected_models=["m1", "m2"],
        weights={"m1": 0.25, "m2": 0.25},
        scenario_id="scenario-1",
        path_id="path-1",
    )

    # 旧 WeightedBlender 不重归一化：预测为 [5, 10]，MAE=(5+10)/2=7.5。
    assert result["status"] == "ok"
    assert result["reference_mode"] == "shared_prediction_matrix"
    assert result["horizon"] == 6
    assert result["n_test"] == 2
    assert result["metrics"]["mae"] == pytest.approx(7.5)
    assert result["models"] == ["m1", "m2"]
    assert result["weights"] == {"m1": 0.25, "m2": 0.25}


def test_system_a_reference_is_invalid_when_a_selected_model_is_not_in_shared_matrix():
    """防止缺候选时沿用部分权重并把降级输出标成有效 System A。"""
    df_test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h"),
            "y": [10.0, 20.0],
            "m1": [12.0, 18.0],
        }
    )

    result = shadow.evaluate_system_a_matrix_reference(
        dataset="pjm",
        horizon=24,
        df_test=df_test,
        selected_models=["m1", "m2"],
        weights={"m1": 0.5, "m2": 0.5},
        scenario_id="scenario-1",
        path_id="path-1",
    )

    assert result["status"] == "invalid_reference"
    assert result["missing_models"] == ["m2"]
    assert "shared prediction matrix" in result["reason"]


def test_pipeline_system_a_selection_accepts_an_explicit_shared_candidate_pool(tmp_path):
    """防止 runner 虽共享预测值，却仍让 System A 从矩阵外模型中选择。"""
    class ScenarioAnalyzer:
        def extract_scenario_signature(self, region_data, region_type):
            return {"mean_load": 10.0}

        def find_similar_scenarios(self, *args, **kwargs):
            return []

    class Combinator:
        def set_historical_scenarios(self, scenarios):
            self.scenarios = scenarios

    class Graph:
        class Nodes:
            def has_node(self, node):
                return False

        G = Nodes()

        def add_scenario_node(self, *args, **kwargs):
            pass

        def instantiate_path(self, *args, **kwargs):
            pass

        def add_scenario_path_edge(self, *args, **kwargs):
            pass

    pipeline = PowerPredictionPipeline.__new__(PowerPredictionPipeline)
    pipeline.scenario_analyzer = ScenarioAnalyzer()
    pipeline.model_combinator = Combinator()
    pipeline.historical_scenarios = []
    pipeline.enable_phase3 = False
    pipeline._infer_region_type = lambda region: "urban"
    pipeline._combinator_trace_path = lambda scenario_id: tmp_path / "trace.json"
    captured = {}

    def select_with_solver(**kwargs):
        captured["available_models"] = kwargs["available_models"]
        class Trace:
            stages = []

        return {
            "models": ["m1"],
            "weights": {"m1": 1.0},
            "path_id": "single_m1",
            "strategy": "single",
            "metrics": {},
        }, Trace()

    pipeline._select_path_with_solver = select_with_solver
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "region": ["R1"] * 3,
            "load": [10.0, 11.0, 12.0],
        }
    )

    selected, weights, _, _ = pipeline.select_models_for_region(
        "R1", train, Graph(), available_models_override=["m1", "m2"]
    )

    assert captured["available_models"] == ["m1", "m2"]
    assert selected == ["m1"]
    assert weights == {"m1": 1.0}


def test_combinator_reference_selects_and_scores_without_training_models(monkeypatch, tmp_path):
    """防止正式参考再次调用 fit_and_predict_region 形成环境相关的部分训练。"""
    feature_root = tmp_path / "features"
    dataset_root = feature_root / "pjm"
    dataset_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "region": ["PJME"] * 3,
            "load": [10.0, 11.0, 12.0],
        }
    ).to_csv(dataset_root / "train.csv", index=False)
    shared_test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-02-01", periods=2, freq="h"),
            "y": [10.0, 20.0],
            "m1": [11.0, 19.0],
        }
    )

    class FakePipeline:
        last_override = None

        def __init__(self):
            self.historical_scenarios = []

        def build_model_graph(self):
            return object()

        def select_models_for_region(
            self, region, train, graph, available_models_override=None
        ):
            FakePipeline.last_override = list(available_models_override or [])
            return ["m1"], {"m1": 1.0}, "scenario-1", "path-1"

        def fit_and_predict_region(self, *args, **kwargs):
            raise AssertionError("shared-matrix System A must not train models")

    monkeypatch.setattr("src.pipeline.main.PowerPredictionPipeline", FakePipeline)

    result = shadow.run_combinator_reference(
        dataset="pjm",
        horizon=6,
        feature_root=feature_root,
        tmpdir=tmp_path / "isolated",
        shared_test_matrix=shared_test,
        shared_models=["m1"],
    )

    assert result["status"] == "ok"
    assert result["metrics"]["mae"] == pytest.approx(1.0)
    assert result["horizon"] == 6
    assert result["reference_mode"] == "shared_prediction_matrix"
    assert FakePipeline.last_override == ["m1"]


def test_candidate_outcome_audit_distinguishes_unavailable_from_filtered_models():
    """防止报告继续把缺失预测文件写成 failed_models={}。"""
    audit = shadow.build_candidate_outcome_audit(
        {
            "safe_models": ["catboost_reg"],
            "eligible_filter_reasons": {
                "catboost_reg": ["alignment_ok", "eligible_pass_all_gates"],
                "arima": ["missing_pred_file_val_or_test"],
                "prophet": ["alignment_ok", "safe_cols_threshold:12.0>10.0"],
            },
        }
    )

    assert audit == {
        "failed_models": {"arima": ["missing_pred_file_val_or_test"]},
        "filtered_models": {
            "prophet": ["alignment_ok", "safe_cols_threshold:12.0>10.0"]
        },
    }


def test_protocol_b_summary_distinguishes_candidate_interaction_from_final_fallback():
    raw = {
        "protocol": "B_fallback_to_best_single_guard",
        "test": {
            "selected_models": ["m1"],
            "weights": {"m1": 1.0},
            "weight_meta": {
                "interaction_branch_candidate": {
                    "enabled": True,
                    "applied": True,
                    "val_mae_raw": 10.0,
                    "val_mae_interaction": 9.0,
                },
                "protocol_b_guard": {
                    "fallback_target": "best_single",
                    "reason": "complexity_guard",
                },
            },
        },
    }

    summary = shadow._protocol_b_split_summary(raw, {"test": [1.0]})

    assert summary["interaction_evaluated"] is True
    assert summary["interaction_candidate_applied"] is True
    assert summary["final_prediction_contains_interaction"] is False
    assert summary["interaction_status_reason"] == "guard_fallback:best_single"
    assert summary["val_mae_raw"] == 10.0


def test_protocol_b_summary_marks_applied_interaction_in_final_nonfallback_prediction():
    raw = {
        "protocol": "B_pred_features",
        "test": {
            "selected_models": ["m1", "m2"],
            "weights": {"m1": 0.5, "m2": 0.5},
            "weight_meta": {
                "interaction_branch": {"enabled": True, "applied": True},
                "protocol_b_guard": {"fallback_target": None, "reason": None},
            },
        },
    }

    summary = shadow._protocol_b_split_summary(raw, {"test": [1.0]})

    assert summary["interaction_evaluated"] is True
    assert summary["interaction_candidate_applied"] is True
    assert summary["final_prediction_contains_interaction"] is True
    assert summary["interaction_status_reason"] == "applied_to_final_prediction"


def test_protocol_b_summary_marks_interaction_overwritten_by_post_adjustment():
    """防止已知的线性后处理覆盖 interaction 后仍误报“最终包含交互项”。"""
    raw = {
        "protocol": "B_pred_features",
        "test": {
            "selected_models": ["m1", "m2"],
            "weights": {"m1": 0.6, "m2": 0.4},
            "weight_meta": {
                "interaction_branch": {"enabled": True, "applied": True},
                "post_adjustment": {"applied": True},
                "protocol_b_guard": {"fallback_target": None, "reason": None},
            },
        },
    }

    summary = shadow._protocol_b_split_summary(raw, {"test": [1.0]})

    assert summary["interaction_candidate_applied"] is True
    assert summary["post_adjustment_applied"] is True
    assert summary["final_prediction_contains_interaction"] is False
    assert summary["interaction_status_reason"] == "overwritten_by_post_adjustment"


def _task(dataset: str, horizon: int, *, b_mae: float = 10.0) -> dict:
    matrix_sha = f"sha-{dataset}-h{horizon}"
    return {
        "status": "ok",
        # §11#7 关系证据门槛：成功任务须记录实际消费到的关系边
        "relation_strength_edges_found": ["lgbm_reg"],
        "dataset": dataset,
        "horizon": horizon,
        "n_val": 100,
        "test_mae_on": b_mae,
        "test_mae_off": 10.2,
        "test_mae_delta": b_mae - 10.2,
        "matrix": {
            "data_sha_test": matrix_sha,
            "n_test": 50,
            "safe_models": ["m1", "m2"],
        },
        "system_a": {
            "status": "ok",
            "reference_mode": "shared_prediction_matrix",
            "dataset": dataset,
            "horizon": horizon,
            "data_sha_test": matrix_sha,
            "n_test": 50,
            "models": ["m1"],
            "metrics": {"mae": 12.0, "rmse": 15.0},
        },
        "protocol_b": {"on": {"mae": b_mae, "rmse": 13.0}},
        "best_single": {
            "model": "m1",
            "selection_uses_test_labels": False,
            "test": {"mae": 10.1, "rmse": 12.0},
        },
    }


def _nine_tasks() -> list[dict]:
    return [
        _task(dataset, horizon)
        for dataset in ("pjm", "aemo_vic", "aemo_nsw")
        for horizon in (1, 6, 24)
    ]


def _write_traces(root: Path) -> None:
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            task_dir = root / dataset / f"h{horizon}"
            task_dir.mkdir(parents=True)
            for mode in ("on", "off"):
                (task_dir / f"protocol_b_trace_{mode}.json").write_text(
                    json.dumps({"stages": [{"stage": "combine"}]}),
                    encoding="utf-8",
                )


def test_quality_gates_pass_only_when_metrics_and_all_18_traces_are_valid(tmp_path):
    """防止 schema 完整被误写成质量门槛通过。"""
    _write_traces(tmp_path)

    gates = shadow.evaluate_quality_gates(_nine_tasks(), trace_root=tmp_path)

    assert gates["status"] == "passed"
    assert gates["system_a_references_valid"]["passed"] is True
    assert gates["best_single_references_valid"]["passed"] is True
    assert gates["average_vs_system_a_1pct"]["passed"] is True
    assert gates["per_task_vs_system_a_3pct"]["passed"] is True
    assert gates["per_task_vs_best_single_1pct"]["passed"] is True
    assert gates["numeric_consistency"]["passed"] is True
    assert gates["trace_integrity"] == {
        "passed": True,
        "expected": 18,
        "valid": 18,
        "issues": [],
    }


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    [
        ("invalid_system_a", "system_a_references_valid"),
        ("matrix_hash_mismatch", "system_a_references_valid"),
        ("best_single_uses_test", "best_single_references_valid"),
        ("wrong_delta", "numeric_consistency"),
        ("system_a_threshold", "average_vs_system_a_1pct"),
        ("best_single_threshold", "per_task_vs_best_single_1pct"),
    ],
)
def test_quality_gates_reject_invalid_references_and_inconsistent_metrics(
    tmp_path, mutation, failed_gate
):
    """每个变异体对应一个曾能被旧 validator 放过的真实缺口。"""
    tasks = _nine_tasks()
    _write_traces(tmp_path)
    if mutation == "invalid_system_a":
        tasks[0]["system_a"] = {"status": "invalid_reference", "reason": "missing model"}
    elif mutation == "matrix_hash_mismatch":
        tasks[0]["system_a"]["data_sha_test"] = "different-matrix"
    elif mutation == "best_single_uses_test":
        tasks[0]["best_single"]["selection_uses_test_labels"] = True
    elif mutation == "wrong_delta":
        tasks[0]["test_mae_delta"] = 999.0
    elif mutation == "system_a_threshold":
        for task in tasks:
            task["system_a"]["metrics"]["mae"] = 9.0
    elif mutation == "best_single_threshold":
        tasks[0]["best_single"]["test"]["mae"] = 9.0

    gates = shadow.evaluate_quality_gates(tasks, trace_root=tmp_path)

    assert gates["status"] == "failed"
    assert gates[failed_gate]["passed"] is False


def test_quality_gates_reject_a_missing_or_malformed_trace(tmp_path):
    tasks = _nine_tasks()
    _write_traces(tmp_path)
    missing = tmp_path / "aemo_nsw" / "h24" / "protocol_b_trace_off.json"
    missing.unlink()
    malformed = tmp_path / "pjm" / "h1" / "protocol_b_trace_on.json"
    malformed.write_text("not-json", encoding="utf-8")

    gates = shadow.evaluate_quality_gates(tasks, trace_root=tmp_path)

    assert gates["status"] == "failed"
    assert gates["trace_integrity"]["passed"] is False
    assert gates["trace_integrity"]["valid"] == 16
    assert len(gates["trace_integrity"]["issues"]) == 2


# --- v6 核心门槛（Task 8.3 Task 7） -----------------------------------------


def _add_pair_references(tasks):
    for task in tasks:
        b_mae = task["protocol_b"]["on"]["mae"]
        task["best_pair_reference"] = {
            "models": ["m1", "m2"],
            "validation_mae": 9.0,
            "selection_uses_test_labels": False,
            "test": {"mae": b_mae},
        }
        task["best_pair_same_as_protocol_b"] = True
        task["explicit_conflict_edges_consumed"] = 0
    return tasks


def _write_v6_traces(root: Path) -> None:
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            task_dir = root / dataset / f"h{horizon}"
            task_dir.mkdir(parents=True)
            for mode in ("relation_warmup", "on", "off", "relation_neutral"):
                (task_dir / f"protocol_b_trace_{mode}.json").write_text(
                    json.dumps({"stages": [{
                        "stage": "ProtocolBBackend",
                        "outputs": {"selection_flow": {
                            "selector_output": ["m1", "m2"],
                            "final_selected_before_fit": ["m1", "m2"],
                        }},
                    }]}),
                    encoding="utf-8",
                )


def test_pair_gate_catches_regression_that_best_single_allows(tmp_path):
    tasks = _add_pair_references(_nine_tasks())
    _write_v6_traces(tmp_path)
    task = tasks[0]
    task["protocol_b"]["on"]["mae"] = 518.0
    task["best_single"]["test"]["mae"] = 586.0
    task["best_pair_reference"] = {
        "models": ["catboost_reg", "lgbm_reg"],
        "validation_mae": 250.0,
        "selection_uses_test_labels": False,
        "test": {"mae": 347.0},
    }
    gates = shadow.evaluate_v6_core_gates(tasks, trace_root=tmp_path)
    assert gates["per_task_vs_best_pair_3pct"]["passed"] is False


def test_relation_qualification_uses_relative_delta_and_win_loss():
    tasks = []
    for index in range(9):
        neutral = 100.0 if index < 3 else 10.0
        relative = -0.01 if index < 5 else 0.005
        tasks.append({
            "status": "ok", "n_val": 100 + index,
            "test_mae_on": neutral * (1.0 + relative),
            "test_mae_relation_neutral": neutral,
        })
    qualification = shadow.evaluate_relation_qualification(tasks)
    assert qualification["mean_relative_delta"] <= 0
    assert qualification["wins"] >= qualification["losses"]
    assert qualification["status"] == "qualified"


def test_relation_failure_does_not_fail_core_chain(tmp_path):
    tasks = _add_pair_references(_nine_tasks())
    _write_v6_traces(tmp_path)
    core = shadow.evaluate_v6_core_gates(tasks, trace_root=tmp_path)
    relation = shadow.evaluate_relation_qualification([
        {**task, "test_mae_on": 11.0, "test_mae_relation_neutral": 10.0}
        for task in tasks
    ])
    assert core["status"] == "passed"
    assert relation["status"] == "unqualified"


def test_run_task_injects_candidate_diagnostic_fields(monkeypatch, tmp_path):
    """走真实 run_task 入口：候选诊断字段必须注入到任务记录。"""
    ts_val = pd.date_range("2026-01-01", periods=20, freq="h")
    ts_test = pd.date_range("2026-02-01", periods=10, freq="h")
    df_val = pd.DataFrame({"timestamp": ts_val, "y": np.ones(20), "m1": np.ones(20), "m2": np.ones(20)})
    df_test = pd.DataFrame({"timestamp": ts_test, "y": np.ones(10), "m1": np.ones(10), "m2": np.ones(10)})
    matrix = {
        "df_val_kg": df_val, "df_test_kg": df_test,
        "df_raw_val": None, "df_raw_test": None,
        "safe_models": ["m1", "m2"],
        "base_model_cols": ["m1", "m2"],
        "metadata": {
            "filter": {}, "eligible_filter_reasons": {"m1": [], "m2": []},
            "frozen_naive": {"loaded": False},
            "common_base_models": ["m1", "m2"],
            "raw": {"val_loaded": True, "test_loaded": True},
        },
    }
    raw = {
        "protocol": "B_pred_features",
        "val": {
            "weight_meta": {
                "protocol_b_selection_meta": {"explicit_conflict_edges_consumed": 2},
            },
        },
        "test": {
            "selected_models": ["m1", "m2"],
            "weights": {"m1": 0.5, "m2": 0.5},
            "weight_meta": {
                "protocol_b_guard": {"fallback_target": None, "reason": None},
                "guard_config": {},
                "interaction_branch": {
                    "applied": True, "enabled": True,
                    "val_mae_raw": 9.0, "val_mae_interaction": 8.5,
                    "cv_mae_raw_guard": 9.4, "cv_mae_interaction_guard": 8.9,
                    "cv_guard_source": "oof_blocked_cv", "cv_oof_coverage": 0.8,
                },
                "post_adjustment": {"applied": False},
            },
        },
    }
    pred = {"test": np.ones(10)}
    trace = SimpleNamespace(stages=[{"stage": "ProtocolBBackend"}])

    monkeypatch.setattr(shadow, "build_task_matrix", lambda **kwargs: matrix)
    monkeypatch.setattr(shadow, "_run_protocol_b_on_matrix", lambda **kwargs: (raw, pred, trace))
    monkeypatch.setattr(shadow, "_run_protocol_a_on_matrix", lambda *a, **k: {"mae": 10.0, "rmse": 12.0})

    candidate_diagnostic = {
        ("pjm", 1): {
            "best_pair": {
                "models": ["m1", "m2"], "validation_mae": 8.0,
                "selection_uses_test_labels": False, "selection_source": "validation_mae_only",
                "test": {"mae": 9.0},
            },
            "best_pair_same_as_protocol_b": True,
            "explicit_conflict_edges_consumed": 2,
        },
    }
    record = shadow.run_task(
        dataset="pjm", horizon=1, models=["m1", "m2"],
        pred_root=tmp_path, raw_root=None, out_root=tmp_path,
        filter_threshold=0.0, seed=42, run_combinator=False,
        candidate_diagnostic=candidate_diagnostic,
    )
    assert record["status"] == "ok"
    assert record["best_pair_reference"]["models"] == ["m1", "m2"]
    assert record["best_pair_reference"]["test"]["mae"] == 9.0
    assert record["best_pair_same_as_protocol_b"] is True
    assert record["explicit_conflict_edges_consumed"] == 2


def test_main_requires_candidate_diagnostic_for_full_run(monkeypatch, tmp_path):
    """完整九任务运行前必须强制 candidate diagnostic，失败立即停止，不先跑完九任务。"""
    import sys

    monkeypatch.setattr(sys, "argv", [
        "run_system_ab_shadow.py",
        "--pred-root", str(tmp_path),
        "--raw-root", str(tmp_path),
        "--feature-root", str(tmp_path),
        "--output", str(tmp_path / "report.json"),
        "--skip-combinator",
    ])
    with pytest.raises(SystemExit) as exc:
        shadow.main()
    assert exc.value.code == 2


def test_main_assembles_v6_report_through_cli(monkeypatch, tmp_path):
    """走 main() 完整报告组装路径：CLI 预检 → 9 任务 → v6 quality_gates +
    relation_qualification → validate_shadow_report 复核通过（main 返回 0）。

    这补上此前只测 run_task / evaluate_v6_core_gates 单点、未经过 CLI 与最终
    report 结构的缺口。
    """
    import sys

    from tests.test_system_ab_shadow_report import _task as _v5_task

    records = []
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            task = _v5_task(dataset=dataset, horizon=horizon, n_val=100 + horizon)
            b_mae = task["test_mae_on"]
            task["best_pair_reference"] = {
                "models": ["m1", "m2"],
                "validation_mae": 9.0,
                "selection_uses_test_labels": False,
                "test": {"mae": b_mae},
            }
            task["best_pair_same_as_protocol_b"] = True
            task["explicit_conflict_edges_consumed"] = 0
            records.append(task)

    out_root = tmp_path / "out"
    _write_v6_traces(out_root)

    diag_path = tmp_path / "candidate_diagnostic.json"
    diag_path.write_text(
        json.dumps({
            "schema_version": "task83-candidate-diagnostic.1",
            "tasks": [
                {
                    "task_id": f"{dataset}_h{horizon}",
                    "dataset": dataset,
                    "horizon": horizon,
                    "matrix_hashes": {"df_val_kg": "x", "df_test_kg": "x"},
                    "safe_models": ["m1", "m2"],
                    "singles": [],
                    "pairs": [
                        {"models": ["m1", "m2"], "validation_mae": 9.0,
                         "test_mae": 10.0, "eligible_pair": True},
                    ],
                    "best_single": {},
                    "best_pair": {
                        "models": ["m1", "m2"],
                        "validation_mae": 9.0,
                        "selection_uses_test_labels": False,
                        "selection_source": "validation_mae_only",
                        "test": {"mae": 10.0},
                    },
                    "protocol_b": {"models": ["m1", "m2"]},
                    "best_pair_same_as_protocol_b": True,
                    "explicit_conflict_edges_consumed": 0,
                }
                for dataset in ("pjm", "aemo_vic", "aemo_nsw")
                for horizon in (1, 6, 24)
            ],
        }),
        encoding="utf-8",
    )

    import scripts.run_protocol_b_candidate_diagnostic as diag_mod
    import scripts.train_baselines as tb

    monkeypatch.setattr(shadow, "_build_kg_model_candidates", lambda: ["m1", "m2"])
    monkeypatch.setattr(
        diag_mod, "verify_locked_sources",
        lambda pred_root, pipeline_config, feature_root: {"status": "verified"},
    )
    monkeypatch.setattr(
        tb, "load_verified_baseline_provenance",
        lambda pred_root, pipeline_config: {"pipeline_sha256": "x", "data_hashes": {}},
    )
    monkeypatch.setattr(shadow, "run_task", lambda **kwargs: records.pop(0))

    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "run_system_ab_shadow.py",
        "--pred-root", str(tmp_path),
        "--raw-root", str(tmp_path),
        "--feature-root", str(tmp_path),
        "--output", str(output),
        "--skip-combinator",
        "--candidate-diagnostic", str(diag_path),
        "--require-baseline-provenance",
        "--seed", "42",
        "--out-root", str(out_root),
    ])

    exit_code = shadow.main()

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "task8-v6.1"
    gates = report["quality_gates"]
    assert gates["status"] == "passed"
    assert gates["trace_integrity"]["expected"] == 36
    assert "per_task_vs_best_pair_3pct" in gates
    assert report["relation_qualification"]["status"] == "qualified"
    assert report["_meta"]["candidate_diagnostic_sha256"]


def _v6_records():
    from tests.test_system_ab_shadow_report import _task as _v5_task

    records = []
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            task = _v5_task(dataset=dataset, horizon=horizon, n_val=100 + horizon)
            b_mae = task["test_mae_on"]
            task["best_pair_reference"] = {
                "models": ["m1", "m2"],
                "validation_mae": 9.0,
                "selection_uses_test_labels": False,
                "test": {"mae": b_mae},
            }
            task["best_pair_same_as_protocol_b"] = True
            task["explicit_conflict_edges_consumed"] = 0
            records.append(task)
    return records


def _v6_report(trace_root):
    records = _v6_records()
    _write_v6_traces(trace_root)
    return {
        "schema_version": "task8-v6.1",
        "task_specs": shadow.build_task_specs(),
        "tasks": records,
        "aggregates": shadow.aggregate_summary(records),
        "quality_gates": shadow.evaluate_v6_core_gates(records, trace_root=trace_root),
        "relation_qualification": shadow.evaluate_relation_qualification(records),
        "_meta": {
            "code_commit": "test",
            "random_seed": 42,
            "data_hashes": {"pjm": "a", "aemo_vic": "b", "aemo_nsw": "c"},
            "python": "/tmp/python",
            "out_root": str(trace_root),
            "environment": {dep: "1.0" for dep in shadow.KEY_DEPENDENCIES},
            "candidate_diagnostic_sha256": "abc",
        },
    }


def test_v6_core_gates_reject_empty_selection_flow(tmp_path):
    """空 selection_flow 或缺 selector_output 的 trace 必须让 trace 门槛失败。"""
    tasks = _add_pair_references(_nine_tasks())
    _write_v6_traces(tmp_path)
    bad = tmp_path / "pjm" / "h1" / "protocol_b_trace_on.json"
    bad.write_text(json.dumps({
        "stages": [{"stage": "ProtocolBBackend", "outputs": {"selection_flow": {}}}],
    }), encoding="utf-8")

    gates = shadow.evaluate_v6_core_gates(tasks, trace_root=tmp_path)

    assert gates["trace_integrity"]["passed"] is False
    assert any("empty_selection_flow" in issue for issue in gates["trace_integrity"]["issues"])


def test_validate_shadow_report_recomputes_relation_qualification(tmp_path):
    """validate 必须重算 relation_qualification，记录值与重算不符时拒绝。"""
    report = _v6_report(tmp_path)
    assert report["relation_qualification"]["status"] == "qualified"
    report["relation_qualification"] = {
        **report["relation_qualification"],
        "status": "unqualified",
    }

    with pytest.raises(ValueError, match="relation_qualification"):
        shadow.validate_shadow_report(report, trace_root=tmp_path)
