"""`run_dataset()` 的训练集合必须由配置决定（seasonal_naive 缺席的根因回归）。

**背景**：`abf2c29` 只往 `configs/pipeline.yaml` 的 `models:` 段补了
`seasonal_naive`，却没改训练脚本。`scripts/train_baselines.py::run_dataset`
内部是**硬编码的 7 个模型列表**，`model_params` 只用来取参数、不决定训练谁。
因此配置改了也没用：baselines_v4b 的 artifacts 仍是 63 条 = 7 模型 × 9 任务，
日志里搜不到 seasonal，九任务基线里 seasonal naive 实际缺席。

**同时暴露的测试盲区**：`tests/test_pipeline_model_params_complete.py` 只校验
"配置里声明了"，全绿却测不到"训练时真的会跑"。本模块补的正是那条集成路径——
断言 `run_dataset` 训练的模型集合等于传入的配置集合，并真实产出
val/test/meta 三个文件与可校验的哈希。

这里用真实模型（seasonal_naive + xgboost_reg，均为秒级），不用假对象：
被测的是"训练集合与产物"，而项目已有"假对象测试全绿、真实路径空转"的教训。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture()
def dataset_root(tmp_path):
    feature_root = tmp_path / "features"
    _write_splits(feature_root)
    return feature_root, tmp_path / "out"


def test_run_dataset_trains_exactly_the_configured_model_set(dataset_root):
    """训练集合必须等于配置集合——不得由脚本内硬编码列表决定。"""
    feature_root, out_root = dataset_root

    results = tb.run_dataset(
        name="pjm",
        feature_root=feature_root,
        target_col="load",
        horizons=[1],
        out_root=out_root,
        max_rows=None,
        model_params=CONFIGURED,
    )

    trained = {rec["model"] for rec in results[1]["artifacts"]}

    assert trained == set(CONFIGURED), (
        f"训练集合与配置不一致：训练了 {sorted(trained)}，配置为 {sorted(CONFIGURED)}。"
        "若出现未配置的模型，说明 run_dataset 仍在用硬编码列表。"
    )


def test_seasonal_naive_artifacts_are_generated_and_hashable(dataset_root):
    """seasonal_naive 必须真实产出 val/test/meta 三个文件，并带可校验哈希。"""
    feature_root, out_root = dataset_root

    results = tb.run_dataset(
        name="pjm",
        feature_root=feature_root,
        target_col="load",
        horizons=[1],
        out_root=out_root,
        max_rows=None,
        model_params=CONFIGURED,
    )

    record = next(r for r in results[1]["artifacts"] if r["model"] == "seasonal_naive")
    expected_names = tb._artifact_file_names(1, "seasonal_naive")

    assert set(record["files"]) == set(expected_names)
    for file_name in expected_names:
        path = out_root / "pjm" / file_name
        assert path.exists(), f"seasonal_naive 未产出 {file_name}"
        assert tb._sha256_file(path) == record["files"][file_name], f"{file_name} 哈希不匹配"

    # 指标也要真实写入，而不是只有文件
    assert "seasonal_naive" in results[1]["val"]
    assert "seasonal_naive" in results[1]["test"]
    assert np.isfinite(results[1]["test"]["seasonal_naive"]["mae"])


def test_provenance_verification_covers_seasonal_naive(dataset_root):
    """逐文件哈希校验必须把 seasonal_naive 一并纳入，缺文件要报错。"""
    feature_root, out_root = dataset_root

    results = tb.run_dataset(
        name="pjm",
        feature_root=feature_root,
        target_col="load",
        horizons=[1],
        out_root=out_root,
        max_rows=None,
        model_params=CONFIGURED,
    )
    provenance = {"artifacts": results[1]["artifacts"]}

    # 正常情况：包含 seasonal_naive 也能通过
    tb.verify_task_artifacts(
        provenance, pred_root=out_root, dataset="pjm", horizon=1,
        models=sorted(CONFIGURED),
    )

    # 篡改其中一个 seasonal_naive 产物后必须被发现
    victim = out_root / "pjm" / tb._artifact_file_names(1, "seasonal_naive")[0]
    victim.write_text(victim.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seasonal_naive"):
        tb.verify_task_artifacts(
            provenance, pred_root=out_root, dataset="pjm", horizon=1,
            models=sorted(CONFIGURED),
        )
