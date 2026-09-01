"""V1 相似度核心对照脚本冒烟测试（scripts/run_v1_similarity_comparison.py）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_v1_similarity_comparison as comparison
import scripts.train_baselines as tb
from tests.test_library_prediction_entry import _build_library


def _write_raw_and_preds(tmp_path: Path, dataset: str, *, val_level: float, test_level: float, n: int = 400) -> None:
    """val 与 test 负荷量级明显不同，用于验证路由签名取自 validation 而非 test。"""
    for split, level, seed in (("val", val_level, 1), ("test", test_level, 2)):
        rng = np.random.default_rng(seed)
        ts = pd.date_range("2025-01-01", periods=n, freq="h")
        temp = np.linspace(4, 30, n) + rng.normal(0, 1, n)
        load = level + 0.1 * level * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 1.5, n)
        raw = pd.DataFrame(
            {"timestamp": ts, "load": load, "hour": ts.hour, "dow": ts.dayofweek, "temp": temp}
        )
        (tmp_path / "features" / dataset).mkdir(parents=True, exist_ok=True)
        raw.to_csv(tmp_path / "features" / dataset / f"{split}.csv", index=False)

        _x, y, ts_target, _freq = tb.prepare_supervised(raw, "load", 24)
        y = np.asarray(y, dtype=float)
        row_ids = tb._build_row_id_from_timestamp(pd.Series(ts_target)).to_numpy()
        (tmp_path / "baselines" / dataset).mkdir(parents=True, exist_ok=True)
        for model, mseed, noise in (("catboost_reg", 11, 2.0), ("lgbm_reg", 22, 3.0)):
            r = np.random.default_rng(mseed)
            pd.DataFrame(
                {"row_id": row_ids, "timestamp": ts_target.values,
                 "pred": y + r.normal(0, noise, len(y)), "y": y}
            ).to_csv(tmp_path / "baselines" / dataset / f"{split}_pred_h24_{model}.csv", index=False)


def test_routing_signature_comes_from_validation_not_test(tmp_path):
    # _build_library: scenario pjm_h24_A (y_mean 100) + aemo_vic_h24_B (y_mean 520)
    lib = _build_library(tmp_path)
    # val 量级 ~100（贴近 pjm_h24_A），test 量级 ~520（贴近 aemo_vic_h24_B）
    _write_raw_and_preds(tmp_path, "pjm", val_level=100.0, test_level=520.0)

    report = comparison.run(
        database=lib["db"],
        pred_root=tmp_path / "baselines",
        raw_root=tmp_path / "features",
        datasets=["pjm"],
        horizons=[24],
    )

    assert report["summary"]["n_tasks"] == 1
    task = report["tasks"][0]
    # 路由必须由 validation 量级决定 -> pjm_h24_A；若误用 test 会命中 aemo_vic_h24_B
    assert task["similarity_match"]["scenario_id"] == "pjm_h24_A"
    assert task["query_signature"]["y_mean"] < 200.0  # 来自 val（~100），不是 test（~520）
    assert task["similarity_match"]["applicable"] is True
    assert task["similarity_match"]["test_mae"] >= 0
    assert 0.0 <= task["similarity_match"]["scenario_similarity_score"] <= 1.0
    assert "ratio_vs_best_single" in task["similarity_match"]
    assert task["no_similarity"]["scenario_id"] in {"pjm_h24_A", "aemo_vic_h24_B"}
    assert isinstance(task["selected_different"], bool)
    assert task["best_single"]["model"] in {"catboost_reg", "lgbm_reg"}
    assert report["_meta"]["commit"]
