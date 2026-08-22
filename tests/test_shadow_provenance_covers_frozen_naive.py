"""正式运行器必须把 frozen seasonal_naive 也纳入来源校验（真实接线，非函数级）。

**已修的上一个缺口**：`16d2c0a` 让 `train_baselines.run_dataset` 按配置训练，
`seasonal_naive` 于是会真实产出 val/test/meta 三件产物。

**本模块要堵的这个缺口**：`verify_task_artifacts` 本身能校验任意模型，但正式
运行器 `run_task` 只把 `metadata["common_base_models"]` 传给它。而
`_build_kg_model_candidates()` 明确把 `seasonal_naive` 从基础候选中剔除
（`src/eval/kg/config.py`：`base = [m for m in MODELS if m != "seasonal_naive"]`），
它是作为 **frozen expert** 单独加载的。因此 `seasonal_naive` 的预测文件
**从未进入来源校验**——旧的、与本轮基线无关的 CSV 可以直接被采信。

上一轮的测试只是手工把 `seasonal_naive` 传给 `verify_task_artifacts`，
证明了"校验函数能查它"，没有证明"正式运行器真的会查它"。这与本项目多次出现的
"函数有效、真实接线空转"同类，故本模块断言打在 `run_task` 上。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

import scripts.run_system_ab_shadow as shadow


class _Recorder:
    def __init__(self):
        self.models_verified: List[str] = []
        self.called = False


def _install(monkeypatch, recorder, *, frozen_loaded: bool):
    """把 run_task 内部重活替身掉，只保留"传给来源校验的模型集合"这条路径。"""

    def fake_build_task_matrix(**kwargs):
        return {
            "df_val_kg": _tiny_frame(),
            "df_test_kg": _tiny_frame(),
            "df_raw_val": None,
            "df_raw_test": None,
            "safe_models": ["xgboost_reg", "lgbm_reg"],
            "base_model_cols": ["xgboost_reg", "lgbm_reg"],
            "metadata": {
                "common_base_models": ["xgboost_reg", "lgbm_reg"],
                "filter": {},
                "eligible_filter_reasons": {},
                # frozen expert 是否真的加载成功，决定它该不该被纳入来源校验
                "frozen_naive": {"enabled": True, "loaded": frozen_loaded},
            },
        }

    def fake_verify(provenance, *, pred_root, dataset, horizon, models):
        recorder.called = True
        recorder.models_verified = list(models)

    monkeypatch.setattr(shadow, "build_task_matrix", fake_build_task_matrix)
    monkeypatch.setattr("scripts.train_baselines.verify_task_artifacts", fake_verify)
    # 校验之后的重活一律短路：本用例只关心校验集合
    monkeypatch.setattr(
        shadow, "build_candidate_outcome_audit", lambda meta: {}, raising=False
    )


def _tiny_frame():
    import numpy as np
    import pandas as pd

    n = 48
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "y": np.linspace(100, 120, n),
        "xgboost_reg": np.linspace(101, 121, n),
        "lgbm_reg": np.linspace(99, 119, n),
    })


def _run(monkeypatch, tmp_path, recorder, *, frozen_loaded: bool):
    _install(monkeypatch, recorder, frozen_loaded=frozen_loaded)
    try:
        shadow.run_task(
            dataset="pjm",
            horizon=1,
            models=["xgboost_reg", "lgbm_reg"],
            pred_root=tmp_path / "pred",
            raw_root=None,
            out_root=tmp_path / "out",
            filter_threshold=2.0,
            seed=42,
            run_combinator=False,
            baseline_provenance={"artifacts": []},
        )
    except Exception:
        # 后续阶段（Protocol A/B 等）失败不影响本用例：校验集合在此之前就已确定
        pass


def test_run_task_verifies_frozen_seasonal_naive_when_it_is_loaded(monkeypatch, tmp_path):
    """frozen seasonal naive 实际加载成功时，必须一并接受来源校验。"""
    recorder = _Recorder()

    _run(monkeypatch, tmp_path, recorder, frozen_loaded=True)

    assert recorder.called, "run_task 未调用来源校验"
    assert "seasonal_naive" in recorder.models_verified, (
        "frozen seasonal_naive 已加载却未纳入 provenance 校验——"
        "其预测文件可以来自与本轮基线无关的旧 CSV"
    )
    # 基础候选仍须全部覆盖
    assert {"xgboost_reg", "lgbm_reg"} <= set(recorder.models_verified)


def test_run_task_does_not_require_seasonal_naive_when_not_loaded(monkeypatch, tmp_path):
    """frozen naive 未加载时不得强行要求它的产物，否则会误判为来源缺失。"""
    recorder = _Recorder()

    _run(monkeypatch, tmp_path, recorder, frozen_loaded=False)

    assert recorder.called
    assert "seasonal_naive" not in recorder.models_verified


def test_verified_model_set_has_no_duplicates(monkeypatch, tmp_path):
    """校验集合不得重复，避免重复校验并保持集合语义。"""
    recorder = _Recorder()

    _run(monkeypatch, tmp_path, recorder, frozen_loaded=True)

    assert len(recorder.models_verified) == len(set(recorder.models_verified))
