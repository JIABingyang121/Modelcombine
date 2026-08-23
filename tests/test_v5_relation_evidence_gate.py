"""v5 必须能证明关系强度真的被消费，否则该轮判为无法评估本功能。

关系强度接入后，若 v5 跑完所有任务的 `relation_strength_edges_found` 都为空，
说明图谱根本没传到（或没有任何关系边），这一轮就**无法评估该功能**——此时
报告不能显示"通过"，否则会得出"接入了但没收益"这种其实无据的结论。

因此新增机器门槛：**至少一个成功任务的 `relation_strength_edges_found` 非空**。
"""
from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_system_ab_shadow as shadow


def _task(dataset: str, horizon: int, edges):
    """构造一条已通过其他门槛的任务记录，只变化关系证据字段。"""
    return {
        "dataset": dataset,
        "horizon": horizon,
        "status": "ok",
        "relation_strength_edges_found": list(edges),
    }


def _all_nine(edges_by_index=None):
    specs = shadow.build_task_specs()
    out = []
    for i, spec in enumerate(specs):
        edges = (edges_by_index or {}).get(i, [])
        out.append(_task(spec["dataset"], spec["horizon"], edges))
    return out


def test_gate_fails_when_no_task_consumed_any_relation_edge():
    """全部任务都没消费到关系边 -> 该门槛必须判失败。"""
    gate = shadow.evaluate_relation_evidence_gate(_all_nine())

    assert gate["passed"] is False
    assert gate["issues"], "失败时必须给出可定位的原因"
    assert gate["tasks_with_edges"] == 0


def test_gate_passes_when_at_least_one_task_consumed_edges():
    gate = shadow.evaluate_relation_evidence_gate(
        _all_nine({0: ["lgbm_reg"], 3: ["xgboost_reg", "catboost_reg"]})
    )

    assert gate["passed"] is True
    assert gate["tasks_with_edges"] == 2


def test_gate_ignores_failed_tasks():
    """只统计成功任务：失败任务没有关系证据是理所当然的。"""
    tasks = _all_nine({0: ["lgbm_reg"]})
    tasks[1]["status"] = "failed"
    tasks[1]["relation_strength_edges_found"] = []

    gate = shadow.evaluate_relation_evidence_gate(tasks)

    assert gate["passed"] is True
    assert gate["tasks_with_edges"] == 1


def test_missing_field_is_treated_as_no_evidence():
    """字段缺失不得被当成通过——旧版报告没有该字段，必须判为无证据。"""
    tasks = _all_nine()
    for t in tasks:
        t.pop("relation_strength_edges_found", None)

    gate = shadow.evaluate_relation_evidence_gate(tasks)

    assert gate["passed"] is False


def test_relation_gate_is_part_of_overall_quality_gates(tmp_path):
    """该门槛必须并入 evaluate_quality_gates，并参与 status 判定。"""
    gates = shadow.evaluate_quality_gates(_all_nine(), trace_root=tmp_path)

    assert "relation_strength_evidence" in gates, (
        "关系证据门槛未并入总门槛——v5 可能在没有任何关系边的情况下显示通过"
    )
    assert gates["relation_strength_evidence"]["passed"] is False
    assert gates["status"] == "failed"
