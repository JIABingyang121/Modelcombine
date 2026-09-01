"""train_baselines.py 真实训练路径写入 SQLite 模型库（SQLite 模型库 Task 3）。

- 提供 --database + --model-artifacts 时：保存拟合后的模型对象并登记 models 行。
- 未提供时：原有 V8 / 离线评估产物与行为完全不变。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.train_baselines as baselines
from src.models.artifacts import load_artifact
from src.storage.model_store import ModelStore

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGURED = {
    "lgbm_reg": {"n_estimators": 5, "n_jobs": 1, "verbose": -1},
    "seasonal_naive": {"seasonal_period": 4},
}


def _write_features(root: Path, n: int = 130) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        r = np.random.default_rng(seed)
        ts = pd.date_range("2026-01-01", periods=n, freq="h")
        load = 100 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24) + r.normal(0, 2, n)
        pd.DataFrame(
            {"timestamp": ts, "load": load, "temp": np.linspace(5, 25, n) + r.normal(0, 1, n)}
        ).to_csv(root / f"{split}.csv", index=False)


def _run_dataset(out: Path, *, store, artifact_dir):
    features = out.parent / "features" / "pjm"
    if not features.exists():
        _write_features(features)
    return baselines.run_dataset(
        "pjm",
        features,
        "load",
        [1],
        out,
        None,
        CONFIGURED,
        model_store=store,
        artifact_dir=artifact_dir,
    )


def test_run_dataset_registers_fitted_model_and_reloadable_artifact(tmp_path):
    db = tmp_path / "lib.sqlite3"
    store = ModelStore(str(db))
    store.create_schema()
    artifact_dir = tmp_path / "artifacts"

    _run_dataset(tmp_path / "out", store=store, artifact_dir=artifact_dir)

    row = store.get_model("pjm__h1__lgbm_reg")
    assert row is not None
    assert row["model_type"] == "lgbm_reg"
    assert row["model_params"] == CONFIGURED["lgbm_reg"]
    # 真实训练特征列（prepare_supervised 之后、reindex 到 train 列）
    assert "temp" in row["required_features"]
    assert "load" not in row["required_features"]
    assert "timestamp" not in row["required_features"]

    artifact_path = Path(row["artifact_path"])
    assert artifact_path.exists()
    model = load_artifact(artifact_path)
    X_eval = pd.DataFrame(
        {c: np.linspace(1, 2, 6) for c in row["required_features"]},
        index=pd.date_range("2027-01-01", periods=6, freq="h"),
    )
    assert len(model.predict(X_eval)) == 6

    assert store.get_model("pjm__h1__seasonal_naive")["model_type"] == "seasonal_naive"
    store.close()


def test_prediction_csv_preserves_saved_model_output_to_1e8(tmp_path):
    features = tmp_path / "features" / "pjm"
    _write_features(features)
    out = tmp_path / "out"
    db = tmp_path / "lib.sqlite3"
    store = ModelStore(str(db))
    store.create_schema()
    config = {
        "xgboost_reg": {
            "n_estimators": 5,
            "max_depth": 2,
            "n_jobs": 1,
            "device": "cpu",
        }
    }

    baselines.run_dataset(
        "pjm",
        features,
        "load",
        [1],
        out,
        None,
        config,
        model_store=store,
        artifact_dir=tmp_path / "artifacts",
    )

    row = store.get_model("pjm__h1__xgboost_reg")
    model = load_artifact(row["artifact_path"])
    test_raw = pd.read_csv(features / "test.csv")
    X_test, _y_test, _ts_test, _freq = baselines.prepare_supervised(test_raw, "load", 1)
    direct = np.asarray(model.predict(X_test[row["required_features"]]), dtype=float)
    saved = pd.read_csv(out / "pjm" / "test_pred_h1_xgboost_reg.csv")["pred"].to_numpy(float)

    np.testing.assert_allclose(saved, direct, rtol=0, atol=1e-8)
    store.close()


def test_without_database_baseline_outputs_are_byte_identical(tmp_path):
    plain = tmp_path / "plain"
    withlib = tmp_path / "withlib"
    _write_features(tmp_path / "features" / "pjm")

    baselines.run_dataset("pjm", tmp_path / "features" / "pjm", "load", [1], plain, None, CONFIGURED)

    db = tmp_path / "lib.sqlite3"
    store = ModelStore(str(db))
    store.create_schema()
    baselines.run_dataset(
        "pjm",
        tmp_path / "features" / "pjm",
        "load",
        [1],
        withlib,
        None,
        CONFIGURED,
        model_store=store,
        artifact_dir=tmp_path / "artifacts",
    )
    store.close()

    plain_files = sorted(p.name for p in plain.rglob("*") if p.is_file())
    withlib_files = sorted(p.name for p in withlib.rglob("*") if p.is_file())
    assert plain_files == withlib_files
    for name in plain_files:
        a = next(plain.rglob(name))
        b = next(withlib.rglob(name))
        assert a.read_bytes() == b.read_bytes(), name


def test_subprocess_train_baselines_populates_model_library(tmp_path):
    features = tmp_path / "features"
    _write_features(features / "pjm")
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        "models:\n"
        "  lgbm_reg:\n    n_estimators: 5\n    n_jobs: 1\n    verbose: -1\n"
        "  seasonal_naive:\n    seasonal_period: 4\n",
        encoding="utf-8",
    )
    db = tmp_path / "lib.sqlite3"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_baselines",
            "--datasets",
            "pjm",
            "--features",
            str(features),
            "--out",
            str(tmp_path / "out"),
            "--pipeline-config",
            str(pipeline),
            "--database",
            str(db),
            "--model-artifacts",
            str(tmp_path / "artifacts"),
            "--allow_partial",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    store = ModelStore(str(db))
    rows = store.connection.execute("SELECT model_id FROM models ORDER BY model_id").fetchall()
    got = {r[0] for r in rows}
    # 3 个 horizon × 2 个模型
    assert got == {
        f"pjm__h{h}__{m}" for h in (1, 6, 24) for m in ("lgbm_reg", "seasonal_naive")
    }
    for model_id, in store.connection.execute("SELECT model_id FROM models"):
        assert Path(store.get_model(model_id)["artifact_path"]).exists()
    store.close()


def test_database_without_artifact_dir_is_rejected(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_baselines",
            "--datasets",
            "pjm",
            "--features",
            str(tmp_path / "features"),
            "--out",
            str(tmp_path / "out"),
            "--database",
            str(tmp_path / "lib.sqlite3"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "model-artifacts" in (proc.stderr + proc.stdout)
