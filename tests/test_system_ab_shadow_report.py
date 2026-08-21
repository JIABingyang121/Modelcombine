"""Task 7：九任务影子对照报告的聚合口径与 schema 测试。

这些测试固定的是 Task 7 Step 0 预先写定的口径：
- 9 个任务 = PJM/AEMO VIC/AEMO NSW × h=1/6/24；
- 逐任务明细、等权平均、按样本量加权平均、interaction 胜/负任务数四种口径缺一不可；
- 每任务必须记录 interaction 开/关的 val/OOF/test 三段差值与覆盖率；
- 每任务还必须记录 System A/combinator、Protocol A、Protocol B、最佳单模型等参考。
"""
from __future__ import annotations

import json
import numpy as np
import pytest

import scripts.run_system_ab_shadow as shadow


def _task(
    *,
    dataset: str = "pjm",
    horizon: int = 1,
    n_val: int = 100,
    test_mae_on: float = 10.0,
    test_mae_off: float = 10.2,
    interaction_applied: bool = True,
) -> dict:
    return {
        "status": "ok",
        "dataset": dataset,
        "horizon": horizon,
        "n_val": n_val,
        "interaction_applied": interaction_applied,
        "interaction_evaluated": True,
        "interaction_candidate_applied": interaction_applied,
        "final_prediction_contains_interaction": interaction_applied,
        "post_adjustment_applied": False,
        "interaction_status_reason": "applied_to_final_prediction",
        "val_mae_raw": 9.0,
        "val_mae_interaction": 8.5,
        "val_mae_delta": -0.5,
        "oof_mae_raw": 9.4,
        "oof_mae_interaction": 8.9,
        "oof_mae_delta": -0.5,
        "cv_oof_coverage": 0.8,
        "test_mae_on": test_mae_on,
        "test_mae_off": test_mae_off,
        "test_mae_delta": test_mae_on - test_mae_off,
        "protocol": "kg_protocol_b",
        "fallback_target": None,
        "selected_models": ["m1", "m2"],
        "matrix": {
            "data_sha_test": f"sha-{dataset}-h{horizon}",
            "n_test": 50,
            "safe_models": ["m1", "m2"],
        },
        "system_a": {
            "status": "ok",
            "reference_mode": "shared_prediction_matrix",
            "dataset": dataset,
            "horizon": horizon,
            "data_sha_test": f"sha-{dataset}-h{horizon}",
            "n_test": 50,
            "models": ["m1", "m2"],
            "metrics": {"mae": 12.0, "rmse": 15.0},
        },
        "protocol_a": {"mae": 10.0, "rmse": 12.5},
        "protocol_b": {
            "on": {"mae": test_mae_on, "rmse": 13.0},
            "off": {"mae": test_mae_off, "rmse": 13.2},
        },
        "best_single": {
            "model": "m1",
            "selection_uses_test_labels": False,
            "test": {"mae": 10.0, "rmse": 12.0},
        },
        "guard": {"status": "ok", "reason": None},
        "selection_diff": {"system_a_vs_protocol_b": "changed"},
        "timing": {"total_sec": 1.0, "matrix_sec": 0.5, "protocol_b_on_sec": 0.2, "protocol_b_off_sec": 0.2},
    }


def _write_traces(root) -> None:
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            task_dir = root / dataset / f"h{horizon}"
            task_dir.mkdir(parents=True, exist_ok=True)
            for mode in ("on", "off"):
                (task_dir / f"protocol_b_trace_{mode}.json").write_text(
                    json.dumps({"stages": [{"stage": "combine"}]}), encoding="utf-8"
                )


def _valid_report(trace_root) -> dict:
    _write_traces(trace_root)
    tasks = []
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            tasks.append(_task(dataset=dataset, horizon=horizon, n_val=100 + horizon))
    report = {
        "schema_version": shadow.REPORT_SCHEMA_VERSION,
        "task_specs": shadow.build_task_specs(),
        "tasks": tasks,
        "aggregates": shadow.aggregate_summary(tasks),
        "_meta": {
            "code_commit": "7d13cc5",
            "random_seed": 42,
            "data_hashes": {"pjm": "abc", "aemo_vic": "def", "aemo_nsw": "ghi"},
            "python": "/tmp/venv/bin/python",
            "out_root": str(trace_root),
            "environment": {
                "numpy": "1.26.0",
                "pandas": "2.0.0",
                "scikit-learn": "1.4.0",
                "xgboost": "2.0.0",
                "lightgbm": "4.0.0",
                "catboost": "1.2.0",
                "torch": "2.0.0",
                "pytorch-lightning": "2.0.0",
                "prophet": "1.1.0",
                "statsmodels": "0.14.0",
                "cmdstanpy": "1.2.4",
            },
        },
    }
    report["quality_gates"] = shadow.evaluate_quality_gates(tasks, trace_root=trace_root)
    return report


def test_task_specs_are_exactly_nine_fixed_tasks():
    specs = shadow.build_task_specs()

    assert specs == [
        {"dataset": "pjm", "horizon": 1},
        {"dataset": "pjm", "horizon": 6},
        {"dataset": "pjm", "horizon": 24},
        {"dataset": "aemo_vic", "horizon": 1},
        {"dataset": "aemo_vic", "horizon": 6},
        {"dataset": "aemo_vic", "horizon": 24},
        {"dataset": "aemo_nsw", "horizon": 1},
        {"dataset": "aemo_nsw", "horizon": 6},
        {"dataset": "aemo_nsw", "horizon": 24},
    ]
    assert len(specs) == 9


def test_validate_report_rejects_missing_task(tmp_path):
    report = _valid_report(tmp_path)
    report["tasks"] = report["tasks"][:-1]
    report["aggregates"] = shadow.aggregate_summary(report["tasks"])

    with pytest.raises(ValueError, match="9"):
        shadow.validate_shadow_report(report)


def test_validate_report_requires_per_task_interaction_fields(tmp_path):
    required = [
        "interaction_applied",
        "interaction_evaluated",
        "interaction_candidate_applied",
        "final_prediction_contains_interaction",
        "post_adjustment_applied",
        "interaction_status_reason",
        "val_mae_raw",
        "val_mae_interaction",
        "val_mae_delta",
        "oof_mae_raw",
        "oof_mae_interaction",
        "oof_mae_delta",
        "cv_oof_coverage",
        "test_mae_on",
        "test_mae_off",
        "test_mae_delta",
        "n_val",
        "horizon",
        "dataset",
        "protocol",
        "fallback_target",
        "selected_models",
    ]
    report = _valid_report(tmp_path)
    task = report["tasks"][0]
    for field in required:
        bad_report = _valid_report(tmp_path)
        bad_report["tasks"][0] = {k: v for k, v in task.items() if k != field}
        bad_report["aggregates"] = shadow.aggregate_summary(bad_report["tasks"])
        with pytest.raises(ValueError, match=f"({field}|task key)"):
            shadow.validate_shadow_report(bad_report)


def test_validate_report_requires_system_reference_fields(tmp_path):
    required = [
        "system_a",
        "protocol_a",
        "protocol_b",
        "best_single",
        "guard",
        "selection_diff",
    ]
    report = _valid_report(tmp_path)
    task = report["tasks"][0]
    for field in required:
        bad_report = _valid_report(tmp_path)
        bad_report["tasks"][0] = {k: v for k, v in task.items() if k != field}
        bad_report["aggregates"] = shadow.aggregate_summary(bad_report["tasks"])
        with pytest.raises(ValueError, match=field):
            shadow.validate_shadow_report(bad_report)


def test_validate_report_requires_meta_commit_seed_and_environment(tmp_path):
    report = _valid_report(tmp_path)
    for field in ("code_commit", "random_seed", "data_hashes", "python", "environment"):
        bad_report = _valid_report(tmp_path)
        bad_report["_meta"] = {k: v for k, v in report["_meta"].items() if k != field}
        with pytest.raises(ValueError, match=field):
            shadow.validate_shadow_report(bad_report)


def test_validate_report_requires_key_dependency_versions(tmp_path):
    required_deps = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "torch",
        "pytorch-lightning",
        "prophet",
        "statsmodels",
        "cmdstanpy",
    ]
    report = _valid_report(tmp_path)
    for dep in required_deps:
        bad_report = _valid_report(tmp_path)
        bad_report["_meta"]["environment"] = {
            k: v for k, v in report["_meta"]["environment"].items() if k != dep
        }
        with pytest.raises(ValueError, match=dep):
            shadow.validate_shadow_report(bad_report)


def test_aggregate_summary_computes_four_required_views():
    tasks = [
        _task(dataset="pjm", horizon=1, n_val=100, test_mae_on=10.0, test_mae_off=10.2),
        _task(dataset="pjm", horizon=6, n_val=200, test_mae_on=20.0, test_mae_off=19.0),
        _task(dataset="aemo_vic", horizon=1, n_val=300, test_mae_on=30.0, test_mae_off=31.0),
    ]

    agg = shadow.aggregate_summary(tasks)

    assert len(agg["task_details"]) == 3
    # 等权平均
    assert agg["equal_weight_average"]["test_mae_on"] == pytest.approx(20.0)
    # 按样本量加权平均
    expected_weighted_on = (10.0 * 100 + 20.0 * 200 + 30.0 * 300) / 600
    assert agg["sample_weighted_average"]["test_mae_on"] == pytest.approx(expected_weighted_on)
    # interaction 胜负任务数：test_delta<0 为胜（interaction 更优）
    assert agg["interaction_win_loss_count"] == {
        "interaction_wins": 2,
        "interaction_losses": 1,
        "interaction_ties": 0,
        "total_tasks": 3,
    }


def test_aggregate_summary_treats_float_ties_as_ties():
    tasks = [
        _task(dataset="pjm", horizon=1, n_val=100, test_mae_on=10.0, test_mae_off=10.0),
    ]
    agg = shadow.aggregate_summary(tasks)
    assert agg["interaction_win_loss_count"]["interaction_ties"] == 1


def test_validate_report_rejects_failed_or_stale_quality_gates(tmp_path):
    report = _valid_report(tmp_path)
    report["tasks"][0]["test_mae_delta"] = 999.0

    with pytest.raises(ValueError, match="(aggregates|quality_gates)"):
        shadow.validate_shadow_report(report, trace_root=tmp_path)


def test_validate_report_rejects_trace_removed_after_gate_was_computed(tmp_path):
    report = _valid_report(tmp_path)
    (tmp_path / "aemo_nsw" / "h24" / "protocol_b_trace_off.json").unlink()

    with pytest.raises(ValueError, match="quality_gates"):
        shadow.validate_shadow_report(report, trace_root=tmp_path)


def test_validate_report_accepts_valid_report(tmp_path):
    shadow.validate_shadow_report(_valid_report(tmp_path), trace_root=tmp_path)
