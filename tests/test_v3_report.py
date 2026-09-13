"""V3 报告脚本的契约测试。

覆盖：主指标任务数、宏平均、胜场计数与规则、3×3 冠军交叉表结构（Modelcombine
列按盲测数据集计算）、T 任务网格强制。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v3_fit_history import DATASET, HORIZONS, POOL, _write_inputs  # noqa: E402

from scripts.v3_fit_history import main as fit_main  # noqa: E402
from scripts.v3_predict_blind import main as blind_main  # noqa: E402
from scripts.v3_report import ReportError  # noqa: E402
from scripts.v3_report import main as report_main  # noqa: E402


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    definition, plan, shared = _write_inputs(tmp_path)
    fit_out = tmp_path / "fit"
    assert fit_main([
        "--definition", str(definition), "--window-plan", str(plan),
        "--shared", str(shared), "--data-root", str(tmp_path / "data"),
        "--out-dir", str(fit_out),
    ]) == 0
    blind_out = tmp_path / "blind"
    assert blind_main([
        "--definition", str(definition), "--window-plan", str(plan),
        "--shared", str(shared), "--fit-dir", str(fit_out),
        "--data-root", str(tmp_path / "data"), "--out-dir", str(blind_out),
    ]) == 0
    return fit_out, blind_out


def _report(tmp_path: Path, fit_out: Path, blind_out: Path, plan: Path, out: Path) -> int:
    return report_main([
        "--definition", str(tmp_path / "definition.json"),
        "--window-plan", str(plan),
        "--predictions", str(blind_out / "predictions.csv"),
        "--fit-dir", str(fit_out),
        "--data-root", str(tmp_path / "data"),
        "--out-dir", str(out),
    ])


def test_report_metrics_and_wins(tmp_path):
    fit_out, blind_out = _prepare(tmp_path)
    out = tmp_path / "report"
    assert _report(tmp_path, fit_out, blind_out, tmp_path / "window_plan.json", out) == 0

    metrics = json.loads((out / "main_metrics.json").read_text(encoding="utf-8"))
    methods = set(POOL) | {"equal_weight", "stacking", "mole_router", "modelcombine"}
    assert set(metrics["macro_wape"]) == methods
    assert len(metrics["tasks"]) == len(methods) * len(HORIZONS)

    wins = json.loads((out / "wins.json").read_text(encoding="utf-8"))
    assert wins["total_tasks"] == len(HORIZONS)
    assert 0 <= wins["full_wins"] <= len(HORIZONS)
    for entry in wins["per_task"]:
        assert entry["best_single"]["model"] in POOL
        assert entry["best_static"]["method"] in {"equal_weight", "stacking"}
        assert entry["dynamic"]["method"] == "mole_router"

    table = json.loads((out / "champion_3x3.json").read_text(encoding="utf-8"))
    assert len(table["table"]) == 1
    row = table["table"][0]
    assert row["test_dataset"] == DATASET
    assert f"{DATASET}_champion" in row
    expected = [
        t["wape"] for t in metrics["tasks"]
        if t["dataset"] == DATASET and t["method"] == "modelcombine"
    ]
    assert row["modelcombine"] == pytest.approx(sum(expected) / len(expected))

    assert (out / "by_dataset.json").is_file()
    assert (out / "by_forecast_steps.json").is_file()


def test_plan_with_extra_test_task_fails_grid(tmp_path):
    fit_out, blind_out = _prepare(tmp_path)
    plan = tmp_path / "window_plan.json"
    data = json.loads(plan.read_text(encoding="utf-8"))
    data["datasets"][0]["windows"].append(
        {"label": "T2", "role": "test", "dataset": DATASET,
         "forecast_origin": "2023-03-01 00:00:00"}
    )
    plan.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportError) as excinfo:
        _report(tmp_path, fit_out, blind_out, plan, tmp_path / "report")
    assert "T 任务网格" in str(excinfo.value)
