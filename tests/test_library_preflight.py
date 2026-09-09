"""模型库完整性只读预检（`scripts/library_preflight.py`）。

预检的意义在于**在读 T1—T3 之前**拦住不完整的库。所以这里逐个造出"差一点"的库，
确认每一种都被抓住：空库、缺基础模型、缺关系、产物丢失、报告矩阵不齐、报告有重复批次。
"""
from __future__ import annotations

import json

import pytest

from scripts.train_combinations_kg import TIMESTAMP_POLICY
from scripts.library_preflight import (
    LibraryIncomplete,
    assert_library_complete,
    assert_window_plan_unchanged,
)
from src.storage.model_store import ModelStore
from tests.library_fixtures import make_complete_library

DATASETS = ("pjm", "aemo_vic", "aemo_nsw")
STEPS = (24, 168, 720)
CANDIDATES = ("lgbm_reg", "seasonal_naive")


def _check(library, **over):
    kwargs = dict(datasets=DATASETS, forecast_steps=STEPS, candidates=CANDIDATES,
                  base_horizon=1, timestamp_policy=TIMESTAMP_POLICY,
                  library_report=library["report"])
    kwargs.update(over)
    return assert_library_complete(library["database"], **kwargs)


@pytest.fixture
def library(tmp_path):
    return make_complete_library(tmp_path, datasets=DATASETS,
                                 forecast_steps=STEPS, candidates=CANDIDATES)


def test_complete_library_passes(library):
    record = _check(library)
    assert record["models"] == len(DATASETS) * len(CANDIDATES)
    assert record["relations"] == len(DATASETS) * len(STEPS) * 3
    assert record["combinations"] == record["relations"]
    assert record["library_report_tasks"] == record["relations"]


def test_empty_database_is_refused(tmp_path):
    empty = tmp_path / "empty.sqlite3"
    ModelStore(str(empty)).create_schema()
    with pytest.raises(LibraryIncomplete, match="基础模型不符"):
        assert_library_complete(empty, datasets=DATASETS, forecast_steps=STEPS,
                                candidates=CANDIDATES, base_horizon=1,
                                timestamp_policy=TIMESTAMP_POLICY)


def test_missing_database_is_refused(tmp_path):
    with pytest.raises(LibraryIncomplete, match="模型库不存在"):
        assert_library_complete(tmp_path / "nope.sqlite3", datasets=DATASETS,
                                forecast_steps=STEPS, candidates=CANDIDATES, base_horizon=1,
                                timestamp_policy=TIMESTAMP_POLICY)


def test_partially_built_library_is_refused(library):
    """只建了一部分关系的库必须在读 T 之前被拦住。"""
    store = ModelStore(str(library["database"]))
    with store.connection:
        store.connection.execute(
            "DELETE FROM scenario_data_combinations WHERE relation_id IN "
            "(SELECT relation_id FROM scenario_data_combinations LIMIT 4)")
    store.close()
    with pytest.raises(LibraryIncomplete, match="关系数是 23"):
        _check(library)


def test_missing_base_model_artifact_is_refused(library):
    (library["artifacts"] / f"pjm__h1__{CANDIDATES[0]}.pkl").unlink()
    with pytest.raises(LibraryIncomplete, match="基础模型的产物不存在"):
        _check(library)


def test_missing_combination_artifact_is_refused(library):
    next(library["artifacts"].glob("*__combo.pkl")).unlink()
    with pytest.raises(LibraryIncomplete, match="组合器产物不存在"):
        _check(library)


def test_incomplete_report_matrix_is_refused(library):
    report = json.loads(library["report"].read_text())
    report["tasks"] = report["tasks"][:-1]
    library["report"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(LibraryIncomplete, match="建库报告"):
        _check(library)


def test_report_with_duplicate_tasks_is_refused(library):
    report = json.loads(library["report"].read_text())
    report["tasks"][1] = dict(report["tasks"][0])
    library["report"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(LibraryIncomplete, match="重复任务|矩阵不符"):
        _check(library)


def test_window_plan_content_must_match_the_definition(tmp_path):
    windows = [
        {"label": "S1", "history_start": "2026-01-01 00:00:00",
         "history_end": "2026-01-02 00:00:00", "forecast_origin": "2026-01-02 00:00:00",
         "targets": {"24": {"forecast_steps": 24}}},
    ]
    definition = {"datasets": [{"dataset": "pjm", "windows": windows}]}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"datasets": [{"dataset": "pjm", "origins": windows}]}),
                    encoding="utf-8")
    assert assert_window_plan_unchanged(plan, definition) is None

    changed = [dict(windows[0], forecast_origin="2099-01-01 00:00:00")]
    plan.write_text(json.dumps({"datasets": [{"dataset": "pjm", "origins": changed}]}),
                    encoding="utf-8")
    with pytest.raises(LibraryIncomplete, match="forecast_origin"):
        assert_window_plan_unchanged(plan, definition)


# ------------------------------------------- 建库报告的时间戳策略必须形成闸门
def test_report_without_timestamp_policy_is_refused(library):
    """写进报告只是记录；不核对就等于没有闸门。缺失即拒绝，不给默认值。"""
    report = json.loads(library["report"].read_text())
    report.pop("timestamp_policy")
    library["report"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(LibraryIncomplete, match="时间戳策略"):
        _check(library)


def test_report_with_a_different_timestamp_policy_is_refused(library):
    """用旧策略建出来的完整库不得通过——同一段历史在两种策略下切出的序列不同。"""
    report = json.loads(library["report"].read_text())
    report["timestamp_policy"] = "raw_rows_as_is"
    library["report"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(LibraryIncomplete, match="不是用当前策略建的"):
        _check(library)


def test_complete_report_records_the_current_policy(library):
    assert _check(library)["timestamp_policy"] == TIMESTAMP_POLICY
