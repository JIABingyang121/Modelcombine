"""固定九任务候选诊断工具（Task 8.3 Task 6）。

锁定来源校验、validation-only 最佳二模型选择、pair 重合与显式 conflict 统计。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_protocol_b_candidate_diagnostic import (
    select_validation_best_pair,
    summarize_diagnostic_coverage,
    validate_diagnostic_schema,
    verify_sha256,
)


def test_sha_verifier_rejects_one_changed_artifact(tmp_path):
    artifact = tmp_path / "prediction.csv"
    artifact.write_text("yhat\n1.0\n", encoding="utf-8")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    verify_sha256(artifact, expected, label="prediction")
    artifact.write_text("yhat\n2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction.*SHA256"):
        verify_sha256(artifact, expected, label="prediction")


def test_best_pair_is_selected_by_validation_only():
    rows = [
        {"models": ["a", "b"], "validation_mae": 5.0, "test_mae": 100.0, "eligible_pair": True},
        {"models": ["a", "c"], "validation_mae": 6.0, "test_mae": 1.0, "eligible_pair": True},
    ]
    assert select_validation_best_pair(rows)["models"] == ["a", "b"]


def test_degenerate_pair_cannot_be_reference():
    rows = [
        {"models": ["a", "b"], "validation_mae": 1.0, "test_mae": 1.0, "eligible_pair": False},
        {"models": ["a", "c"], "validation_mae": 2.0, "test_mae": 2.0, "eligible_pair": True},
    ]
    assert select_validation_best_pair(rows)["models"] == ["a", "c"]


def test_diagnostic_reports_pair_reference_overlap_and_explicit_conflicts():
    tasks = [
        {"task_id": "d1_h1", "best_pair": {"models": ["a", "b"]},
         "protocol_b": {"models": ["b", "a"]}, "explicit_conflict_edges_consumed": 0},
        {"task_id": "d1_h6", "best_pair": {"models": ["a", "c"]},
         "protocol_b": {"models": ["a", "b"]}, "explicit_conflict_edges_consumed": 1},
    ]
    summary = summarize_diagnostic_coverage(tasks)
    assert summary["best_pair_same_as_protocol_b_count"] == 1
    assert summary["best_pair_same_as_protocol_b_tasks"] == ["d1_h1"]
    assert summary["tasks_with_explicit_conflict_edges"] == 1


def test_default_task_specs_are_consumable_by_diagnostic_builder(monkeypatch):
    """默认 build_task_specs() 必须能直接生成九任务诊断，不得要求手工补 models。"""
    from scripts import run_protocol_b_candidate_diagnostic as diag
    from scripts import run_system_ab_shadow as shadow
    from src.eval.kg import protocol_b as protocol_b_module

    specs = shadow.build_task_specs()
    assert len(specs) == 9
    assert all("models" not in s for s in specs), "build_task_specs 规格本身不应含 models"

    ts_val = pd.date_range("2026-01-01", periods=100, freq="h")
    ts_test = pd.date_range("2026-02-01", periods=50, freq="h")

    def _frame(ts):
        df = pd.DataFrame({"timestamp": ts, "y": np.zeros(len(ts))})
        for m in ("m1", "m2", "m3"):
            df[m] = np.zeros(len(ts))
        return df

    matrix = {
        "df_val_kg": _frame(ts_val),
        "df_test_kg": _frame(ts_test),
        "df_raw_val": pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour}),
        "df_raw_test": pd.DataFrame({"timestamp": ts_test, "hour": ts_test.hour}),
        "safe_models": ["m1", "m2", "m3"],
        "base_model_cols": ["m1", "m2", "m3"],
    }
    monkeypatch.setattr(diag, "verify_locked_sources", lambda **kwargs: {"status": "verified"})
    monkeypatch.setattr(shadow, "build_task_matrix", lambda **kwargs: matrix)
    monkeypatch.setattr(
        protocol_b_module,
        "evaluate_fixed_protocol_b_candidate",
        lambda *a, **k: {
            "val": {"mae": 1.0}, "test": {"mae": 1.0},
            "eligible_pair": True, "degenerate_reason": None,
            "guard_would_fallback_to": None, "guard_would_fallback_reason": None,
        },
    )
    monkeypatch.setattr(
        protocol_b_module,
        "kg_combination_with_features",
        lambda *a, **k: {
            "val": {"selected_models": ["m1", "m2"], "weight_meta": {
                "protocol_b_selection_meta": {"explicit_conflict_edges_consumed": 0},
                "guard_config": {"final_fallback_target": None, "final_fallback_reason": None},
            }},
            "relation_feedback": {"eligible": True, "skip_reason": None, "by_model": {}},
        },
    )

    report = diag.build_diagnostic_report(
        pred_root=Path("/tmp/pred"), raw_root=None,
        feature_root=Path("/tmp/features"), pipeline_config=Path("/tmp/pipeline.yaml"),
        tasks=specs,
    )
    assert len(report["tasks"]) == 9
    assert all(t["task_id"] for t in report["tasks"])


def _schema_task(dataset, horizon, *, best_pair=None, pairs=None):
    default_pairs = [
        {"models": ["m1", "m2"], "validation_mae": 1.0, "test_mae": 1.0, "eligible_pair": True},
    ]
    return {
        "task_id": f"{dataset}_h{horizon}",
        "dataset": dataset, "horizon": horizon,
        "matrix_hashes": {"df_val_kg": "x", "df_test_kg": "x"},
        "safe_models": ["m1", "m2", "m3"],
        "singles": [], "pairs": pairs if pairs is not None else default_pairs,
        "best_single": {},
        "best_pair": best_pair,
        "protocol_b": {"models": ["m1", "m2"]},
        "best_pair_same_as_protocol_b": True,
        "explicit_conflict_edges_consumed": 0,
    }


def _valid_best_pair(models=("m1", "m2"), *, validation_mae=1.0, test_mae=1.0):
    return {
        "models": list(models),
        "validation_mae": validation_mae,
        "selection_uses_test_labels": False,
        "selection_source": "validation_mae_only",
        "test": {"mae": test_mae},
    }


def _nine_unique_tasks():
    tasks = []
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            tasks.append(_schema_task(dataset, horizon, best_pair=_valid_best_pair()))
    return tasks


def _schema_report(tasks):
    return {"schema_version": "task83-candidate-diagnostic.1", "tasks": tasks}


def test_diagnostic_schema_rejects_duplicate_tasks():
    """九条记录必须唯一；重复 (dataset,horizon) 会被字典覆盖，必须拒绝。"""
    tasks = _nine_unique_tasks()
    tasks[8] = _schema_task("pjm", 1, best_pair=_valid_best_pair())  # 重复 pjm_h1
    with pytest.raises(ValueError, match="unique"):
        validate_diagnostic_schema(_schema_report(tasks))


def test_diagnostic_schema_rejects_missing_best_pair():
    """每条都必须存在有效 best pair。"""
    tasks = _nine_unique_tasks()
    tasks[0] = _schema_task("pjm", 1, best_pair=None)
    with pytest.raises(ValueError, match="best_pair"):
        validate_diagnostic_schema(_schema_report(tasks))


def test_diagnostic_schema_rejects_best_pair_not_validation_selected():
    """best_pair 必须是其 pair 列表按 validation 选出的结果。"""
    pairs = [
        {"models": ["m1", "m2"], "validation_mae": 1.0, "eligible_pair": True},
        {"models": ["m1", "m3"], "validation_mae": 0.5, "eligible_pair": True},
    ]
    tasks = _nine_unique_tasks()
    # 把最佳 pair 硬写成 ["m1","m2"]，但按 validation 应选 ["m1","m3"]。
    tasks[0] = _schema_task(
        "pjm", 1, best_pair=_valid_best_pair(("m1", "m2")), pairs=pairs,
    )
    with pytest.raises(ValueError, match="validation-selected"):
        validate_diagnostic_schema(_schema_report(tasks))


def test_diagnostic_schema_accepts_validation_selected_best_pair():
    """正确场景：best_pair 与按 validation 选出的结果一致（含数值）。"""
    pairs = [
        {"models": ["m1", "m2"], "validation_mae": 1.0, "test_mae": 2.0, "eligible_pair": True},
        {"models": ["m1", "m3"], "validation_mae": 0.5, "test_mae": 1.5, "eligible_pair": True},
    ]
    tasks = _nine_unique_tasks()
    tasks[0] = _schema_task(
        "pjm", 1,
        best_pair=_valid_best_pair(("m1", "m3"), validation_mae=0.5, test_mae=1.5),
        pairs=pairs,
    )
    validate_diagnostic_schema(_schema_report(tasks))


def test_diagnostic_schema_rejects_best_pair_validation_mae_mismatch():
    """best_pair.models 正确但 validation_mae 伪造时，必须拒绝。"""
    pairs = [
        {"models": ["m1", "m2"], "validation_mae": 1.0, "test_mae": 2.0, "eligible_pair": True},
    ]
    tasks = _nine_unique_tasks()
    bp = _valid_best_pair(("m1", "m2"), validation_mae=0.5, test_mae=2.0)
    tasks[0] = _schema_task("pjm", 1, best_pair=bp, pairs=pairs)
    with pytest.raises(ValueError, match="validation_mae"):
        validate_diagnostic_schema(_schema_report(tasks))


def test_diagnostic_schema_rejects_best_pair_test_mae_mismatch():
    """best_pair.test.mae 是 pair 门槛分母，伪造时必须拒绝。"""
    pairs = [
        {"models": ["m1", "m2"], "validation_mae": 1.0, "test_mae": 2.0, "eligible_pair": True},
    ]
    tasks = _nine_unique_tasks()
    bp = _valid_best_pair(("m1", "m2"), validation_mae=1.0, test_mae=0.1)
    tasks[0] = _schema_task("pjm", 1, best_pair=bp, pairs=pairs)
    with pytest.raises(ValueError, match="test.mae"):
        validate_diagnostic_schema(_schema_report(tasks))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_diagnostic_schema_rejects_non_finite_best_pair_metrics(bad):
    """NaN/Inf 会使 abs() 比较恒为 False，必须显式拒绝非有限数值。"""
    pairs = [
        {"models": ["m1", "m2"], "validation_mae": bad, "test_mae": bad, "eligible_pair": True},
    ]
    tasks = _nine_unique_tasks()
    bp = _valid_best_pair(("m1", "m2"), validation_mae=bad, test_mae=bad)
    tasks[0] = _schema_task("pjm", 1, best_pair=bp, pairs=pairs)
    with pytest.raises(ValueError, match="finite"):
        validate_diagnostic_schema(_schema_report(tasks))
