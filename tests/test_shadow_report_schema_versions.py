"""新增关系证据门槛后，旧版影子报告必须仍然可复核。

`evaluate_quality_gates` 的**结构**变了（多了 relation_strength_evidence），
而 `validate_shadow_report` 会把记录的门槛和重算结果逐字段比对。若不区分
schema 版本，已提交的 v4d（task7-shadow.3）在当前代码下会直接报
`quality_gates do not match recomputed metrics or trace files`——
"v4d 是第一个连续验收版本"这个结论就失去了可复核性。

约定：旧 schema 按旧门槛验证；只有新 schema 才强制关系证据门槛。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_system_ab_shadow as shadow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V4D_DIR = PROJECT_ROOT / "result" / "ab_convergence" / "shadow_9tasks_v4d"


def test_current_schema_version_is_bumped_for_the_new_gate():
    """门槛结构变化必须伴随 schema 升级，否则新旧报告无法区分。"""
    assert shadow.REPORT_SCHEMA_VERSION == "task8-v6.1"
    assert "task7-shadow.3" in shadow.LEGACY_REPORT_SCHEMA_VERSIONS
    assert "task7-shadow.4" in shadow.LEGACY_REPORT_SCHEMA_VERSIONS


def test_legacy_gates_exclude_relation_evidence():
    tasks = [
        {"dataset": s["dataset"], "horizon": s["horizon"], "status": "ok"}
        for s in shadow.build_task_specs()
    ]
    legacy = shadow.evaluate_quality_gates(
        tasks, trace_root=Path("/nonexistent"), include_relation_gate=False
    )
    assert "relation_strength_evidence" not in legacy

    current = shadow.evaluate_quality_gates(tasks, trace_root=Path("/nonexistent"))
    assert "relation_strength_evidence" in current


@pytest.mark.skipif(not (V4D_DIR / "shadow_9tasks.json").exists(), reason="v4d 报告不在仓库中")
def test_committed_v4d_report_still_validates_under_current_code():
    report = json.loads((V4D_DIR / "shadow_9tasks.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "task7-shadow.3"

    # 不得抛异常：旧版验收证据必须在当前代码下仍可复核
    shadow.validate_shadow_report(report, trace_root=V4D_DIR / "traces")


def test_new_schema_report_without_relation_evidence_is_rejected():
    """新 schema 下缺关系证据必须判失败，不能靠旧口径蒙混。"""
    report = {
        "schema_version": shadow.REPORT_SCHEMA_VERSION,
        "task_specs": shadow.build_task_specs(),
        "tasks": [],
    }
    with pytest.raises(ValueError):
        shadow.validate_shadow_report(report, trace_root=Path("/nonexistent"))
