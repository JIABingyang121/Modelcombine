"""seasonal_naive 的 `val_pred_*.meta.json` sidecar 必须让真实矩阵接线放行它。

**本模块要堵的缺口**：`16d2c0a` 让 `run_dataset` 真正训练 seasonal_naive（产出
val/test 预测 CSV），`dafd998` 让来源校验覆盖 frozen seasonal_naive。但
`train_baselines` 从不生成扩展候选安全门要求的 `val_pred_h*_seasonal_naive.meta.json`。
于是 `src/eval/kg/data_io.py::_load_extended_pool_for_split` 把 `val_eval_mode`
判为 `unknown`，在 `allow_in_sample=False` 的 strict 模式下拦截 seasonal_naive，
导致 `frozen_naive.loaded=false`——预测文件明明存在却进不了矩阵。

本模块直接跑真实的 `run_dataset → build_task_matrix` 路径，断言 sidecar 存在且
`val_eval_mode=deterministic` 时 frozen seasonal_naive 真的被加载。这与本项目
"函数有效、真实接线空转"的教训同源：`run_itransformer_adapter` 早就为 itransformer
写 sidecar，但基线训练路径漏掉了 seasonal_naive。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_system_ab_shadow as shadow
import scripts.train_baselines as tb


def _write_splits(root: Path, n: int = 260) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split, start in (("train", "2026-01-01"), ("val", "2026-02-01"), ("test", "2026-03-01")):
        ts = pd.date_range(start, periods=n, freq="h")
        rng = np.random.default_rng(abs(hash(split)) % 1000)
        df = pd.DataFrame({
            "timestamp": ts,
            "load": 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 1.0, n),
            "hour": ts.hour,
            "dow": ts.dayofweek,
            "lag_1": np.linspace(99, 199, n),
        })
        df.to_csv(root / f"{split}.csv", index=False)


CONFIGURED = {
    "seasonal_naive": {"seasonal_period": 24},
    "xgboost_reg": {"n_estimators": 20, "max_depth": 3, "random_state": 42, "n_jobs": 1},
}


def _train(tmp_path: Path) -> Path:
    feature_root = tmp_path / "features"
    out_root = tmp_path / "pred"
    _write_splits(feature_root)
    tb.run_dataset(
        name="pjm",
        feature_root=feature_root,
        target_col="load",
        horizons=[1],
        out_root=out_root,
        max_rows=None,
        model_params=CONFIGURED,
    )
    return out_root


def test_run_dataset_then_build_task_matrix_loads_frozen_seasonal_naive(tmp_path):
    """真实 run_dataset 产出后，build_task_matrix 必须真正加载 frozen seasonal_naive。"""
    out_root = _train(tmp_path)

    meta_path = out_root / "pjm" / "val_pred_h1_seasonal_naive.meta.json"
    assert meta_path.exists(), "train_baselines 未生成 seasonal_naive 的 val sidecar"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload.get("val_eval_mode") == "deterministic", (
        "sidecar 的 val_eval_mode 必须是 deterministic，否则会被扩展候选安全门拦截"
    )

    matrix = shadow.build_task_matrix(
        dataset="pjm",
        horizon=1,
        models=["xgboost_reg"],
        pred_root=out_root,
        raw_root=None,
    )

    frozen = matrix["metadata"]["frozen_naive"]
    assert frozen.get("loaded") is True, (
        f"seasonal_naive 未通过真实矩阵接线加载（loaded=false）：{frozen}"
    )
