"""v6 两轮验收验证器（Task 8.3 Task 7）。

两轮报告一致性、模型集合、数值与 selection_flow 的一致性断言。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_protocol_b_v6 as v6


def _run(models, *, test_mae_on=1.0):
    return {
        "tasks": [
            {"dataset": "pjm", "horizon": 1, "selected_models": list(models),
             "test_mae_on": test_mae_on, "weights": {"m1": 0.5, "m2": 0.5}},
        ]
    }


def test_assert_same_task_models_passes_for_identical_runs():
    run = _run(["m1", "m2"])
    v6.assert_same_task_models(run, run)


def test_assert_same_task_models_rejects_different_models():
    with pytest.raises(ValueError, match="model"):
        v6.assert_same_task_models(_run(["m1", "m2"]), _run(["m2", "m3"]))


def test_assert_numeric_close_accepts_within_tolerance():
    a = _run(["m1", "m2"], test_mae_on=1.0)
    b = _run(["m1", "m2"], test_mae_on=1.0 + 1e-9)
    v6.assert_numeric_close(a, b, fields=("weights", "test_mae_on"), atol=1e-8)


def test_assert_numeric_close_rejects_drift_beyond_tolerance():
    a = _run(["m1", "m2"], test_mae_on=1.0)
    b = _run(["m1", "m2"], test_mae_on=1.1)
    with pytest.raises(ValueError):
        v6.assert_numeric_close(a, b, fields=("weights", "test_mae_on"), atol=1e-8)


def test_assert_same_locked_sources_compares_data_hashes():
    a = {"_meta": {"data_hashes": {"pjm": "abc"}}}
    b = {"_meta": {"data_hashes": {"pjm": "abc"}}}
    v6.assert_same_locked_sources(a, b)


def test_assert_same_locked_sources_rejects_mismatch():
    a = {"_meta": {"data_hashes": {"pjm": "abc"}}}
    b = {"_meta": {"data_hashes": {"pjm": "def"}}}
    with pytest.raises(ValueError):
        v6.assert_same_locked_sources(a, b)


def test_resolve_trace_root_falls_back_when_server_out_root_absent(tmp_path):
    """服务器绝对 out_root 拉回本机后不存在，须回退到报告旁 traces/。"""
    report_path = tmp_path / "report.json"
    (tmp_path / "traces").mkdir()

    resolved = v6._resolve_trace_root(
        {"out_root": "/disk14T_2/byjia/Modelcombine/result/nonexistent"},
        report_path,
    )
    assert resolved == tmp_path / "traces"


def test_resolve_trace_root_keeps_existing_out_root(tmp_path):
    """out_root 仍存在时优先使用它（本机与服务器一致的重算口径）。"""
    report_path = tmp_path / "report.json"
    out_root = tmp_path / "out"
    out_root.mkdir()

    resolved = v6._resolve_trace_root({"out_root": str(out_root)}, report_path)
    assert resolved == out_root


# --- 第三轮后续复审：pair 基准 vs 运行时字段的绑定修正 -----------------------


def _pair_task(**overrides):
    base = {
        "dataset": "pjm", "horizon": 1,
        "selected_models": ["m1", "m2"],
        "best_pair_reference": {"models": ["m1", "m2"]},
        "best_pair_same_as_protocol_b": True,
        "explicit_conflict_edges_consumed": 0,
    }
    base.update(overrides)
    return base


def _diag(best_pair_models=("m1", "m2"), same=True, conflict=0):
    return {
        "schema_version": "task83-candidate-diagnostic.1",
        "tasks": [{
            "dataset": "pjm", "horizon": 1,
            "best_pair": {"models": list(best_pair_models)},
            "best_pair_same_as_protocol_b": same,
            "explicit_conflict_edges_consumed": conflict,
        }],
    }


def test_pair_reference_still_requires_diagnostic_equality():
    """best_pair_reference 必须仍等于诊断基准，不能凭空编造。"""
    run = {"tasks": [_pair_task(best_pair_reference={"models": ["m1", "m3"]})]}
    with pytest.raises(ValueError, match="best_pair_reference differs"):
        v6.assert_pair_references_match_diagnostic(run, _diag())


def test_pair_reference_allows_runtime_same_and_conflict_divergence():
    """关系机制改变本轮选择 / 本轮真实触发显式 conflict 时，不得被强制等于静态诊断。"""
    run = {"tasks": [_pair_task(
        selected_models=["m2", "m3"],
        best_pair_same_as_protocol_b=False,
        explicit_conflict_edges_consumed=1,
    )]}
    v6.assert_pair_references_match_diagnostic(run, _diag(same=True, conflict=0))


def test_pair_reference_recomputes_same_from_selected_models():
    """存储的 best_pair_same_as_protocol_b 必须与本轮 selected_models 一致。"""
    run = {"tasks": [_pair_task(
        selected_models=["m2", "m3"],
        best_pair_same_as_protocol_b=True,  # 错误：selected != best_pair
    )]}
    with pytest.raises(ValueError, match="best_pair_same_as_protocol_b"):
        v6.assert_pair_references_match_diagnostic(run, _diag())


def test_runtime_pair_overlap_and_conflict_aggregates_from_tasks():
    """v6_acceptance 的 overlap/conflict 必须从本轮任务汇总，而非静态诊断。"""
    tasks = [
        {"dataset": "pjm", "horizon": 1, "best_pair_same_as_protocol_b": True,
         "explicit_conflict_edges_consumed": 0},
        {"dataset": "pjm", "horizon": 6, "best_pair_same_as_protocol_b": False,
         "explicit_conflict_edges_consumed": 2},
        {"dataset": "aemo_vic", "horizon": 1, "best_pair_same_as_protocol_b": True,
         "explicit_conflict_edges_consumed": 0},
    ]
    result = v6._runtime_pair_overlap_and_conflict(tasks)
    assert result["pair_reference_overlap"] == {"count": 2, "tasks": ["pjm_h1", "aemo_vic_h1"]}
    assert result["explicit_conflict_edges_consumed"] == {"count": 1, "tasks": ["pjm_h6"]}


# --- build_v6_acceptance 端到端：显式 trace root 必须真正生效 -----------------


def _write_v6_traces(root: Path) -> None:
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            task_dir = root / dataset / f"h{horizon}"
            task_dir.mkdir(parents=True, exist_ok=True)
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


def _best_pair(mae=10.0):
    return {
        "models": ["m1", "m2"],
        "validation_mae": 9.0,
        "selection_uses_test_labels": False,
        "selection_source": "validation_mae_only",
        "test": {"mae": mae},
    }


def _full_v6_records():
    from tests.test_system_ab_shadow_report import _task as _v5_task

    records = []
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            t = _v5_task(dataset=dataset, horizon=horizon, n_val=100 + horizon)
            t["best_pair_reference"] = _best_pair(mae=t["test_mae_on"])
            t["best_pair_same_as_protocol_b"] = True
            t["explicit_conflict_edges_consumed"] = 0
            t["protocol_b_on"] = {"weights": {"m1": 0.5, "m2": 0.5}}
            t["protocol_b_off"] = {"weights": {"m1": 0.5, "m2": 0.5}}
            records.append(t)
    return records


def _full_v6_report(trace_root: Path, out_root_meta: str) -> dict:
    import scripts.run_system_ab_shadow as shadow

    records = _full_v6_records()
    _write_v6_traces(trace_root)
    return {
        "schema_version": "task8-v6.1",
        "task_specs": shadow.build_task_specs(),
        "tasks": records,
        "aggregates": shadow.aggregate_summary(records),
        "quality_gates": shadow.evaluate_v6_core_gates(records, trace_root=trace_root),
        "relation_qualification": shadow.evaluate_relation_qualification(records),
        "_meta": {
            "code_commit": "x", "random_seed": 42,
            "data_hashes": {"pjm": "a", "aemo_vic": "b", "aemo_nsw": "c"},
            "python": "/tmp/python", "out_root": out_root_meta,
            "environment": {dep: "1.0" for dep in shadow.KEY_DEPENDENCIES},
            "candidate_diagnostic_sha256": "abc",
        },
    }


def _diagnostic_full():
    tasks = []
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            tasks.append({
                "task_id": f"{dataset}_h{horizon}",
                "dataset": dataset, "horizon": horizon,
                "matrix_hashes": {"df_val_kg": "x", "df_test_kg": "x"},
                "safe_models": ["m1", "m2"],
                "singles": [],
                "pairs": [
                    {"models": ["m1", "m2"], "validation_mae": 9.0,
                     "test_mae": 10.0, "eligible_pair": True},
                ],
                "best_single": {},
                "best_pair": _best_pair(),
                "protocol_b": {"models": ["m1", "m2"]},
                "best_pair_same_as_protocol_b": True,
                "explicit_conflict_edges_consumed": 0,
            })
    return {"schema_version": "task83-candidate-diagnostic.1", "tasks": tasks}


def test_build_v6_acceptance_passes_explicit_trace_roots(tmp_path):
    """服务器 _meta.out_root 在本机不存在时，显式 trace root 必须让验收通过。"""
    r1_root = tmp_path / "r1"
    r2_root = tmp_path / "r2"
    missing_out_root = str(tmp_path / "server_abs_out_root")  # 不存在

    run1 = _full_v6_report(r1_root, missing_out_root)
    run2 = _full_v6_report(r2_root, missing_out_root)

    acceptance = v6.build_v6_acceptance(
        run1, run2,
        run1_trace_root=r1_root,
        run2_trace_root=r2_root,
        diagnostic=_diagnostic_full(),
    )
    assert acceptance["core_status"] == "passed"
    assert acceptance["relation_qualification_status"] == "qualified"


def _two_runs(tmp_path):
    r1_root = tmp_path / "r1"
    r2_root = tmp_path / "r2"
    missing = str(tmp_path / "server_abs_out_root")
    run1 = _full_v6_report(r1_root, missing)
    run2 = _full_v6_report(r2_root, missing)
    return run1, run2, r1_root, r2_root


def test_assert_pair_runtime_consistent_rejects_divergence():
    """两轮 pair 重合与显式 conflict 消费必须逐任务一致，否则汇总 run1 不可信。"""
    def run(same, conflict):
        return {"tasks": [{
            "dataset": "pjm", "horizon": 1,
            "best_pair_same_as_protocol_b": same,
            "explicit_conflict_edges_consumed": conflict,
        }]}

    v6.assert_pair_runtime_consistent(run(True, 0), run(True, 0))
    with pytest.raises(ValueError, match="best_pair_same_as_protocol_b differs between runs"):
        v6.assert_pair_runtime_consistent(run(True, 0), run(False, 0))
    with pytest.raises(ValueError, match="explicit_conflict_edges_consumed differs between runs"):
        v6.assert_pair_runtime_consistent(run(True, 0), run(True, 1))


def test_build_v6_acceptance_binds_run2_to_diagnostic(tmp_path):
    """run2 的 best_pair_reference 也必须来自同一诊断，不能只校验 run1。"""
    run1, run2, r1_root, r2_root = _two_runs(tmp_path)
    run2["tasks"][0]["best_pair_reference"] = {
        "models": ["m1", "m3"],
        "validation_mae": 9.0,
        "selection_uses_test_labels": False,
        "selection_source": "validation_mae_only",
        "test": {"mae": 10.0},
    }
    with pytest.raises(ValueError, match="best_pair_reference differs"):
        v6.build_v6_acceptance(
            run1, run2, run1_trace_root=r1_root, run2_trace_root=r2_root,
            diagnostic=_diagnostic_full(),
        )


def test_build_v6_acceptance_rejects_run_pair_runtime_divergence(tmp_path):
    """run2 记录不同的 explicit_conflict_edges_consumed 时验收必须失败。"""
    run1, run2, r1_root, r2_root = _two_runs(tmp_path)
    run2["tasks"][0]["explicit_conflict_edges_consumed"] = 1
    with pytest.raises(ValueError, match="explicit_conflict_edges_consumed differs between runs"):
        v6.build_v6_acceptance(
            run1, run2, run1_trace_root=r1_root, run2_trace_root=r2_root,
            diagnostic=_diagnostic_full(),
        )
