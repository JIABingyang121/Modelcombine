"""V3 盲测脚本的契约测试。

覆盖：全方法 T 预测行数与集合、四项验收通过、Modelcombine 检索记录早于 T 起点。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v3_fit_history import DATASET, HORIZONS, POOL, _write_inputs  # noqa: E402

from scripts.v3_fit_history import main as fit_main  # noqa: E402
from scripts.v3_predict_blind import main as blind_main  # noqa: E402


def test_blind_predictions_and_acceptance(tmp_path):
    definition, plan, shared = _write_inputs(tmp_path)
    fit_out = tmp_path / "fit"
    assert fit_main([
        "--definition", str(definition),
        "--window-plan", str(plan),
        "--shared", str(shared),
        "--data-root", str(tmp_path / "data"),
        "--out-dir", str(fit_out),
    ]) == 0

    blind_out = tmp_path / "blind"
    assert blind_main([
        "--definition", str(definition),
        "--window-plan", str(plan),
        "--shared", str(shared),
        "--fit-dir", str(fit_out),
        "--data-root", str(tmp_path / "data"),
        "--out-dir", str(blind_out),
    ]) == 0

    predictions = pd.read_csv(blind_out / "predictions.csv")
    methods = set(POOL) | {"equal_weight", "stacking", "mole_router", "modelcombine"}
    assert set(predictions["method"]) == methods
    assert len(predictions) == len(methods) * sum(HORIZONS)
    assert predictions["yhat"].notna().all()

    acceptance = json.loads((blind_out / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["passed"] is True
    assert acceptance["checks"]["shared_columns_identical"]["passed"] is True
    assert acceptance["checks"]["complete_finite_nonnegative"]["passed"] is True
    assert acceptance["checks"]["no_truth_read"]["passed"] is True

    retrievals = acceptance["checks"]["memory_before_test_origin"]["retrievals"]
    assert len(retrievals) == len(HORIZONS)
    for entry in retrievals:
        assert pd.Timestamp(entry["origin"]) < pd.Timestamp(entry["test_origin"])
        assert entry["members"]
