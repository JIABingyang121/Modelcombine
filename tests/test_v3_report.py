"""V3 报告脚本的契约测试。

覆盖：主指标任务数、宏平均、胜场计数与规则、3×3 冠军交叉表结构。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v3_fit_history import DATASET, HORIZONS, POOL, _write_inputs  # noqa: E402

from scripts.v3_fit_history import main as fit_main  # noqa: E402
from scripts.v3_predict_blind import main as blind_main  # noqa: E402
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


def test_report_metrics_and_wins(tmp_path):
    fit_out, blind_out = _prepare(tmp_path)
    out = tmp_path / "report"
    assert report_main([
        "--definition", str(tmp_path / "definition.json"),
        "--window-plan", str(tmp_path / "window_plan.json"),
        "--predictions", str(blind_out / "predictions.csv"),
        "--fit-dir", str(fit_out),
        "--data-root", str(tmp_path / "data"),
        "--out-dir", str(out),
    ]) == 0

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
    assert "modelcombine" in row

    assert (out / "by_dataset.json").is_file()
    assert (out / "by_forecast_steps.json").is_file()
