"""interaction 组合的离线/在线特征契约一致性（SQLite 模型库 V1 P0-2）。

真实闭环：真实模型产物加载 -> 基础预测 -> run.py predict -> 与离线最终 test
预测在 1e-8 内一致。fixture 构造 load 依赖 temp、而模型只看 hour/dow，使
残差与 temp 相关，触发 interaction 分支；只有当离线组合评估与在线预测使用
同一份"预测起点特征 X(t)"时，两侧才能对齐。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.train_baselines as tb
from scripts.train_combinations_kg import _forecast_origin_raw_frame, _scenario_signature
from src.models.artifacts import save_artifact
from src.models.registry import model_registry
from src.storage.model_store import ModelStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DS = "pjm"
H = 24
MODEL_FEATURES = ["hour", "dow"]  # 模型不看 temp -> 残差 ~ temp


def _make_dataset(raw_root: Path, n: int = 360) -> None:
    for split, start, seed in (
        ("train", "2025-01-01", 1),
        ("val", "2025-03-01", 2),
        ("test", "2025-06-01", 3),
    ):
        r = np.random.default_rng(seed)
        ts = pd.date_range(start, periods=n, freq="h")
        hour = ts.hour.to_numpy()
        dow = ts.dayofweek.to_numpy()
        temp = 15.0 + 10.0 * np.sin(np.arange(n) * 2 * np.pi / 168) + r.normal(0, 1.5, n)
        load = 120.0 + 20.0 * np.sin(hour * 2 * np.pi / 24) + 3.0 * temp + r.normal(0, 1.0, n)
        (raw_root / DS).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"timestamp": ts, "load": load, "hour": hour, "dow": dow, "temp": temp}
        ).to_csv(raw_root / DS / f"{split}.csv", index=False)


def _fit_and_register(store: ModelStore, raw_root: Path, artifacts: Path, pred_root: Path) -> None:
    x_train, y_train, _ts, _freq = tb.prepare_supervised(
        pd.read_csv(raw_root / DS / "train.csv"), "load", H
    )
    # 名称必须在 _build_kg_model_candidates() 内，load_predictions_safe 才会加载
    for name, mseed in (("catboost_reg", 11), ("lgbm_reg", 22)):
        model = model_registry.create(
            "lgbm_reg", n_estimators=30, n_jobs=1, verbose=-1, random_state=mseed
        )
        model.fit(x_train[MODEL_FEATURES], y_train)
        path = save_artifact(model, artifacts / f"{DS}__h{H}__{name}.pkl")
        store.add_model(
            model_id=f"{DS}__h{H}__{name}",
            model_type=name,
            task_type="load_forecast",
            artifact_path=str(path),
            required_features=MODEL_FEATURES,
            model_params={"n_estimators": 30},
            lifecycle_stage="active",
        )
        for split in ("val", "test"):
            x_s, y_s, ts_s, _f = tb.prepare_supervised(
                pd.read_csv(raw_root / DS / f"{split}.csv"), "load", H
            )
            row_ids = tb._build_row_id_from_timestamp(pd.Series(ts_s.values)).to_numpy()
            pd.DataFrame(
                {
                    "row_id": row_ids,
                    "timestamp": ts_s.values,
                    "pred": model.predict(x_s[MODEL_FEATURES]),
                    "y": y_s.values,
                }
            ).to_csv(pred_root / DS / f"{split}_pred_h{H}_{name}.csv", index=False)


def test_interaction_combo_replays_through_run_py_predict(tmp_path):
    raw_root = tmp_path / "features"
    pred_root = tmp_path / "baselines"
    artifacts = tmp_path / "model_artifacts"
    db = tmp_path / "lib.sqlite3"
    (pred_root / DS).mkdir(parents=True, exist_ok=True)

    _make_dataset(raw_root)
    store = ModelStore(str(db))
    store.create_schema()
    _fit_and_register(store, raw_root, artifacts, pred_root)
    store.close()

    build = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DS, "--horizons", str(H),
            "--pred-root", str(pred_root), "--raw-root", str(raw_root),
            "--out-root", str(tmp_path / "ml"),
            "--database", str(db), "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    task = json.loads((tmp_path / "ml" / "model_library_report.json").read_text())["tasks"][0]
    assert task["has_interaction"] is True, "fixture 未触发 interaction，测试失去意义"

    # 在线：forecast-origin 特征（X(t) + 目标时间戳），不含未来 y
    x_test, _y, ts_test, freq = tb.prepare_supervised(
        pd.read_csv(raw_root / DS / "test.csv"), "load", H
    )
    features = x_test.reset_index(drop=True)
    features.insert(0, "timestamp", ts_test.values)
    features.to_csv(tmp_path / "future.csv", index=False)

    # 场景签名来自 validation（不读 test 标签）
    _xv, yv, tsv, _fv = tb.prepare_supervised(pd.read_csv(raw_root / DS / "val.csv"), "load", H)
    signature = _scenario_signature(
        pd.DataFrame({"timestamp": tsv.values, "y": np.asarray(yv, dtype=float)}),
        _forecast_origin_raw_frame(raw_root, DS, "val", H),
        H,
        freq,
    )
    (tmp_path / "scenario.json").write_text(
        json.dumps(
            {
                "task_type": "load_forecast",
                "business_domain": "power_load",
                "region": DS,
                "horizon": H,
                "freq": freq,
                "signature": signature,
            }
        ),
        encoding="utf-8",
    )

    predict = subprocess.run(
        [
            sys.executable, "run.py", "predict",
            "--database", str(db),
            "--scenario", str(tmp_path / "scenario.json"),
            "--features", str(tmp_path / "future.csv"),
            "--output", str(tmp_path / "online.csv"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert predict.returncode == 0, predict.stdout + predict.stderr

    online = pd.read_csv(tmp_path / "online.csv")["yhat"].to_numpy(dtype=float)
    offline = np.asarray(task["test_prediction"], dtype=float)
    assert len(online) == len(offline)
    np.testing.assert_allclose(online, offline, rtol=0, atol=1e-8)
