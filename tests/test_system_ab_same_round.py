"""Task 6A：System A/B 同轮质量对照的口径与报告测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.prediction_pool import RegionPredictionBundle

import scripts.compare_system_ab_same_round as same_round


def _bundle() -> RegionPredictionBundle:
    ts_val = pd.date_range("2026-01-01", periods=4, freq="h")
    ts_test = pd.date_range("2026-01-02", periods=4, freq="h")
    return RegionPredictionBundle(
        df_val=pd.DataFrame(
            {
                "timestamp": ts_val,
                "y": [10.0, 20.0, 30.0, 40.0],
                "m_val": [10.0, 20.0, 30.0, 40.0],
                "m_test": [11.0, 21.0, 31.0, 41.0],
                "informer": [50.0, 50.0, 50.0, 50.0],
            }
        ),
        df_test=pd.DataFrame(
            {
                "timestamp": ts_test,
                "y": [15.0, 25.0, 35.0, 45.0],
                "m_val": [17.0, 27.0, 37.0, 47.0],
                "m_test": [15.0, 25.0, 35.0, 45.0],
                "informer": [50.0, 50.0, 50.0, 50.0],
            }
        ),
        df_raw_val=pd.DataFrame({"timestamp": ts_val, "hour": [0, 1, 2, 3]}),
        df_raw_test=pd.DataFrame({"timestamp": ts_test, "hour": [0, 1, 2, 3]}),
        model_cols=["m_val", "m_test", "informer"],
        base_model_cols=["m_val", "m_test", "informer"],
        fitted_test_models={"m_val": object(), "m_test": object(), "informer": object()},
        metadata={"failed_models": {}, "region": "R"},
    )


def test_metric_summary_uses_one_shared_definition():
    metrics = same_round.metric_summary(
        np.asarray([1.0, 3.0]),
        np.asarray([2.0, 5.0]),
    )

    assert metrics == {"mae": 1.5, "rmse": np.sqrt(2.5)}


def test_best_single_reports_validation_selected_and_test_oracle_separately():
    result = same_round.best_single_summaries(_bundle())

    assert result["validation_selected"]["model"] == "m_val"
    assert result["validation_selected"]["validation_mae"] == 0.0
    assert result["validation_selected"]["test"]["mae"] == 2.0
    assert result["test_oracle"]["model"] == "m_test"
    assert result["test_oracle"]["test"]["mae"] == 0.0
    assert result["test_oracle"]["selection_uses_test_labels"] is True


def test_traditional_pool_excludes_only_declared_deep_models():
    bundle = _bundle()

    traditional = same_round.traditional_model_cols(bundle.model_cols)
    subset = same_round.subset_bundle(bundle, traditional)

    assert traditional == ["m_val", "m_test"]
    assert subset.model_cols == ["m_val", "m_test"]
    assert list(subset.df_val.columns) == ["timestamp", "y", "m_val", "m_test"]
    assert list(subset.df_test.columns) == ["timestamp", "y", "m_val", "m_test"]
    assert bundle.model_cols == ["m_val", "m_test", "informer"]


def test_remove_unstable_late_only_keeps_other_rejection_reasons():
    filter_ctx = {
        "model_cols": ["stable"],
        "dedup_removed": [],
        "stability_removed": {
            "late_only": "unstable_late",
            "late_and_bad": "unstable_late,worse_than_naive",
        },
        "error_corrs": {("stable", "late_only"): 0.1},
    }

    updated, reinstated = same_round.remove_unstable_late_only(
        filter_ctx,
        original_model_cols=["stable", "late_only", "late_and_bad"],
    )

    assert updated["model_cols"] == ["stable", "late_only"]
    assert updated["stability_removed"] == {
        "late_and_bad": "unstable_late,worse_than_naive"
    }
    assert reinstated == ["late_only"]


def test_quality_gate_requires_protocol_b_to_match_both_references():
    passed = same_round.quality_gate(
        protocol_b_mae=100.5,
        combinator_mae=100.0,
        validation_selected_single_mae=100.0,
        tolerance_ratio=1.01,
    )
    failed = same_round.quality_gate(
        protocol_b_mae=102.0,
        combinator_mae=100.0,
        validation_selected_single_mae=100.0,
        tolerance_ratio=1.01,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["vs_combinator"]["passed"] is False
    assert failed["vs_validation_selected_single"]["passed"] is False


def test_report_schema_distinguishes_facts_inferences_and_unknowns():
    report = {
        "schema_version": same_round.REPORT_SCHEMA_VERSION,
        "rows": "720",
        "repeat": 2,
        "python": "/tmp/venv/bin/python",
        "runs": [
            {
                "status": "ok",
                "input": {"data_sha": "abc", "n_train": 648, "n_test": 72},
                "combinator": {"metrics": {"mae": 1.0, "rmse": 1.0}},
                "protocol_a": {"metrics": {"mae": 1.0, "rmse": 1.0}},
                "protocol_b": {"metrics": {"mae": 1.0, "rmse": 1.0}},
                "best_single": {"validation_selected": {}, "test_oracle": {}},
                "quality_gate": {"passed": True},
            }
        ],
        "conclusions": {
            "observed_facts": ["fact"],
            "evidence_supported_inferences": ["inference"],
            "still_unknown": ["unknown"],
        },
        "guarded_state_before": {},
        "guarded_state_after": {},
        "readonly_guarantee_held": True,
    }

    same_round.validate_report(report)


def test_repeat_worker_command_uses_same_interpreter_and_separate_output(tmp_path):
    command = same_round.worker_command(
        rows="720",
        run_index=2,
        output_path=tmp_path / "run2.json",
    )

    assert command[0] == same_round.sys.executable
    assert command[1] == str(same_round.Path(same_round.__file__).resolve())
    assert "--worker-run-index" in command
    assert command[command.index("--worker-run-index") + 1] == "2"
    assert command[command.index("--worker-output") + 1].endswith("run2.json")
