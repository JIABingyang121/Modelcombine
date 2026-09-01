"""离线不限成员数量的最佳组合构建（SQLite 模型库 Task 4）。

通过真实 scripts/train_combinations_kg.py --model-library 入口运行：
- validation 目标明确偏好三个模型的任务 -> SQLite 组合恰好三成员；
- 另一个任务最佳组合恰好两成员；
- test 标签改变不改变已由 validation 选出的成员；
- 数据库保存的组合器重放结果与脚本报告一致（容差 1e-8）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_baselines import _build_row_id_from_timestamp
from src.eval.combination_utils import load_predictions_safe
from src.models.artifacts import load_artifact
from src.storage.model_store import ModelStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = "pjm"
GOOD = ["catboost_reg", "lgbm_reg", "xgboost_reg"]
TASK_N = 460  # 原始序列长度；预测/目标序列为 n - horizon


def test_prediction_loader_orders_duplicate_timestamps_by_row_id(tmp_path):
    pred_root = tmp_path / "baselines"
    task_root = pred_root / DATASET
    task_root.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "row_id": ["2026-11-01 02:00:00_1", "2026-11-01 02:00:00_0", "2026-11-01 03:00:00_0"],
            "timestamp": ["2026-11-01 02:00:00", "2026-11-01 02:00:00", "2026-11-01 03:00:00"],
            "pred": [20.0, 10.0, 30.0],
            "y": [2.0, 1.0, 3.0],
        }
    )
    for model in ("catboost_reg", "lgbm_reg"):
        rows.to_csv(task_root / f"test_pred_h24_{model}.csv", index=False)

    loaded = load_predictions_safe(
        pred_root, DATASET, 24, ["catboost_reg", "lgbm_reg"], "test"
    )

    assert loaded["row_id"].tolist() == [
        "2026-11-01 02:00:00_0",
        "2026-11-01 02:00:00_1",
        "2026-11-01 03:00:00_0",
    ]
    assert loaded["catboost_reg"].tolist() == [10.0, 20.0, 30.0]


def _y_for_split(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (
        100.0
        + 12.0 * np.sin(2 * np.pi * t / 24)
        + 3.0 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.5, n)
    )


def _write_task(pred_root: Path, raw_root: Path, horizon: int, *, third_model: bool,
                n: int = TASK_N, y_test_override=None) -> None:
    for split, yseed, noiseseed in (("val", 1, 10), ("test", 2, 20)):
        ts = pd.date_range("2026-01-01", periods=n, freq="h")
        load = _y_for_split(n, yseed)
        (raw_root / DATASET).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"timestamp": ts, "load": load, "hour": ts.hour, "temp": np.linspace(5, 25, n)}
        ).to_csv(raw_root / DATASET / f"{split}.csv", index=False)

        # 目标序列 = load.shift(-horizon)，预测 CSV 用目标时间戳与匹配的 row_id
        target = load[horizon:].copy()
        target_ts = ts[horizon:]
        if split == "test" and y_test_override is not None:
            target = np.asarray(y_test_override, dtype=float)
        rng = np.random.default_rng(noiseseed + horizon)
        # 独立同质噪声模型：完整 validation 目标明确偏好其平均（方差缩减真实）。
        preds = {
            "catboost_reg": target + rng.normal(0, 3.0, len(target)),
            "lgbm_reg": target + rng.normal(0, 3.0, len(target)),
        }
        if third_model:
            preds["xgboost_reg"] = target + rng.normal(0, 3.0, len(target))

        row_ids = _build_row_id_from_timestamp(pd.Series(target_ts)).to_numpy()
        (pred_root / DATASET).mkdir(parents=True, exist_ok=True)
        for model, pred in preds.items():
            pd.DataFrame(
                {"row_id": row_ids, "timestamp": target_ts, "pred": pred, "y": target}
            ).to_csv(pred_root / DATASET / f"{split}_pred_h{horizon}_{model}.csv", index=False)


def _seed_models(db: Path) -> None:
    store = ModelStore(str(db))
    store.create_schema()
    for h in (1, 6):
        for m in ("catboost_reg", "lgbm_reg", "xgboost_reg"):
            store.add_model(
                model_id=f"{DATASET}__h{h}__{m}",
                model_type=m,
                task_type="load_forecast",
                artifact_path=f"/artifacts/{DATASET}__h{h}__{m}.pkl",
                required_features=["hour", "temp"],
                model_params={"k": 1},
                lifecycle_stage="active",
            )
    store.close()


def _run_build(tmp_path: Path, *, y_test_override_h6=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pred_root = tmp_path / "baselines"
    raw_root = tmp_path / "features"
    out_root = tmp_path / "out"
    db = tmp_path / "lib.sqlite3"
    if not db.exists():
        _seed_models(db)
    _write_task(pred_root, raw_root, 1, third_model=True)
    _write_task(pred_root, raw_root, 6, third_model=False, y_test_override=y_test_override_h6)

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg",
            "--model-library",
            "--datasets", DATASET,
            "--horizons", "1", "6",
            "--pred-root", str(pred_root),
            "--raw-root", str(raw_root),
            "--out-root", str(out_root),
            "--database", str(db),
            "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((out_root / "model_library_report.json").read_text())
    return report, db, {"pred_root": pred_root, "raw_root": raw_root}


def _task(report, horizon):
    return next(t for t in report["tasks"] if t["horizon"] == horizon)


def test_build_library_task_rejects_non_replayable_combination(tmp_path, monkeypatch):
    import scripts.train_combinations_kg as tck

    pred_root = tmp_path / "baselines"
    raw_root = tmp_path / "features"
    db = tmp_path / "lib.sqlite3"
    _seed_models(db)
    _write_task(pred_root, raw_root, 1, third_model=True)

    store = ModelStore(str(db))
    real_load = tck.load_artifact

    def _perturbing_load(path):
        predictor = real_load(path)
        base_predict = predictor.predict
        predictor.predict = lambda *a, **k: np.asarray(base_predict(*a, **k), dtype=float) + 1.0
        return predictor

    monkeypatch.setattr(tck, "load_artifact", _perturbing_load)
    with pytest.raises(RuntimeError, match="重放误差"):
        tck._build_library_task(
            store,
            dataset=DATASET,
            horizon=1,
            kg_models=["catboost_reg", "lgbm_reg", "xgboost_reg"],
            pred_root=pred_root,
            raw_root=raw_root,
            artifact_dir=tmp_path / "combo_artifacts",
            filter_threshold=2.0,
        )
    # 重放不一致 -> 任何数据库写入都不发生
    for table in ("scenarios", "data_profiles", "combinations", "scenario_data_combinations"):
        assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    store.close()


def test_model_library_build_fails_loudly_on_missing_task_artifacts(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out_root = tmp_path / "out"
    db = tmp_path / "lib.sqlite3"
    store = ModelStore(str(db))
    store.create_schema()
    store.close()

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg",
            "--model-library",
            "--datasets", DATASET,
            "--horizons", "24",  # 未写任何 h=24 预测产物
            "--pred-root", str(tmp_path / "baselines"),
            "--raw-root", str(tmp_path / "features"),
            "--out-root", str(out_root),
            "--database", str(db),
            "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "h=24" in (proc.stdout + proc.stderr)
    assert not (out_root / "model_library_report.json").exists()


def test_validation_decides_member_count_three_vs_two(tmp_path):
    report, db, _ = _run_build(tmp_path)

    h1 = _task(report, 1)
    h6 = _task(report, 6)
    assert sorted(m.split("__")[-1] for m in h1["effective_members"]) == GOOD
    assert len(h1["effective_members"]) == 3
    assert sorted(m.split("__")[-1] for m in h6["effective_members"]) == ["catboost_reg", "lgbm_reg"]
    assert len(h6["effective_members"]) == 2

    store = ModelStore(str(db))
    combo = store.get_combination(h1["combination_id"])
    assert [m["model_id"] for m in combo["members"]] == h1["effective_members"]
    assert store.get_relation(h1["relation_id"])["validation_mae"] == pytest.approx(
        h1["validation_mae"], abs=1e-9
    )
    # 每个 scenario/data_profile 只写一条最终最佳关系
    assert store.connection.execute(
        "SELECT COUNT(*) FROM scenario_data_combinations"
    ).fetchone()[0] == 2
    store.close()


def test_test_label_change_does_not_change_selected_members(tmp_path):
    report_a, db_a, _ = _run_build(tmp_path / "a")
    shuffled = _y_for_split(TASK_N - 6, 7)[::-1].copy()  # 长度 = n - horizon(6)
    report_b, db_b, _ = _run_build(tmp_path / "b", y_test_override_h6=shuffled)

    a6 = _task(report_a, 6)
    b6 = _task(report_b, 6)
    assert a6["effective_members"] == b6["effective_members"]
    # test 标签变了，test_mae 应当不同（证明 override 生效），但成员不变
    assert a6["test_mae"] != pytest.approx(b6["test_mae"], abs=1e-6)


def test_saved_combination_predictor_replays_script_report(tmp_path):
    report, db, roots = _run_build(tmp_path)
    store = ModelStore(str(db))

    for horizon in (1, 6):
        task = _task(report, horizon)
        combo = store.get_combination(task["combination_id"])
        predictor = load_artifact(combo["artifact_path"])

        df_test = pd.read_csv(
            roots["pred_root"] / DATASET / f"test_pred_h{horizon}_catboost_reg.csv"
        )
        raw_test = pd.read_csv(roots["raw_root"] / DATASET / "test.csv")
        base = {}
        member_types = [m.split("__")[-1] for m in predictor.member_ids]
        for mt in member_types:
            col = pd.read_csv(
                roots["pred_root"] / DATASET / f"test_pred_h{horizon}_{mt}.csv"
            )["pred"].to_numpy(dtype=float)
            base[mt] = col

        # predictor.member_ids 是模型类型（xgboost_reg 等），与列名一致
        replay = predictor.predict(base, raw_test)
        y = df_test["y"].to_numpy(dtype=float)
        replay_mae = float(np.mean(np.abs(replay - y)))
        assert replay_mae == pytest.approx(task["test_mae"], abs=1e-8)
        np.testing.assert_allclose(
            replay, np.asarray(task["test_prediction"], dtype=float), rtol=0, atol=1e-8
        )
    store.close()
