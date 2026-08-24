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
