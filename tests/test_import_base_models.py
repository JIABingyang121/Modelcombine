"""基础模型导入入口（`scripts/import_base_models.py`）。

这个入口的存在理由是：旧库的 24 个基础模型在训练截止上仍然合规，但它的 27 条关系是按
另一套窗口建的、必须重建；而直接对旧库再建一次库不会报错，只会把新关系追加到旧关系旁边。
所以要能把 models 行搬进一个全新的空库。

这里守住四件事：训练截止门控在建库之前生效且失败时不留半成品；候选必须齐全；
只搬 models 不搬任何关系；产物缺失要被发现。
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.import_base_models import (
    IMPORT_REPORT_FILENAME,
    RELATION_TABLES,
    ImportError_,
    assert_training_ends_before_s1,
    read_source_models,
    s1_history_starts,
    write_target,
)
from src.storage.model_store import ModelStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("pjm", "aemo_vic", "aemo_nsw")
CANDIDATES = ("lgbm_reg", "seasonal_naive")
S1_START = "2026-03-01 00:00:00"

PIPELINE = (
    "models:\n"
    "  lgbm_reg:\n    n_estimators: 5\n    n_jobs: 1\n"
    "  seasonal_naive:\n    seasonal_period: 4\n"
)


def _write_window_plan(path: Path, s1: str = S1_START) -> Path:
    path.write_text(json.dumps({"datasets": [
        {"dataset": d, "fits": True, "origins": [
            {"label": lb, "history_start": s1, "history_end": s1,
             "forecast_origin": s1, "targets": {}}
            for lb in ("S1", "S2", "S3", "A", "T1", "T2", "T3")
        ]}
        for d in DATASETS
    ]}), encoding="utf-8")
    return path


def _write_train(root: Path, *, end: str, rows: int = 200) -> None:
    """只写 train.csv 的时间列与负荷；导入入口只会读时间列。"""
    for dataset in DATASETS:
        (root / dataset).mkdir(parents=True, exist_ok=True)
        stamps = pd.date_range(end=pd.Timestamp(end), periods=rows, freq="h")
        pd.DataFrame({"timestamp": stamps, "load": range(rows)}).to_csv(
            root / dataset / "train.csv", index=False
        )


def _build_source(tmp_path: Path, *, artifacts: bool = True,
                  models=None, lifecycle: str = "active") -> Path:
    """造一个"旧库"：24 个基础模型 + 一整套关系数据（关系不应被搬走）。"""
    artifact_dir = tmp_path / "old_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "old.sqlite3"
    store = ModelStore(str(source))
    store.create_schema()
    for dataset in DATASETS:
        for model_type in (models if models is not None else CANDIDATES):
            model_id = f"{dataset}__h1__{model_type}"
            path = artifact_dir / f"{model_id}.pkl"
            if artifacts:
                path.write_bytes(b"artifact")
            store.add_model(
                model_id=model_id, model_type=model_type, task_type="load_forecast",
                artifact_path=str(path), required_features=["hour", "lag_1"],
                model_params={"n_estimators": 5}, lifecycle_stage=lifecycle,
            )
    # 旧关系：一条完整链路，导入后目标库里一行都不该有
    store.add_scenario(scenario_id="old_scn", task_type="load_forecast",
                       business_domain="power", region="pjm", horizon=1,
                       forecast_steps=24, freq="h", signature={"k": 1})
    profile_id = store.add_data_profile(
        scenario_id="old_scn", data_ref="/old/pjm", target_column="load",
        features=["hour"], sample_count=24, start_at="2023-01-01",
        end_at="2023-01-02", signature={"k": 1})
    combo = store.add_combination("protocol_b_combination", str(artifact_dir / "c.pkl"),
                                  [(f"pjm__h1__{CANDIDATES[0]}", 0, 1.0)])
    relation = store.add_relation("old_scn", profile_id, combo, validation_mae=1.0)
    store.record_prediction_run(relation, "/old/pred.csv")
    store.close()
    return source


@pytest.fixture
def workspace(tmp_path):
    source = _build_source(tmp_path)
    features = tmp_path / "features"
    _write_train(features, end="2026-02-01 00:00:00")   # 严格早于 S1
    plan = _write_window_plan(tmp_path / "windows.json")
    return {"source": source, "features": features, "plan": plan,
            "target": tmp_path / "new.sqlite3", "out": tmp_path / "out",
            "pipeline": tmp_path / "pipeline.yaml", "tmp": tmp_path}


def _cmd(ws, **over):
    args = {
        "--source": str(ws["source"]), "--target": str(ws["target"]),
        "--features": str(ws["features"]), "--window-plan": str(ws["plan"]),
        "--pipeline-config": str(ws["pipeline"]), "--out": str(ws["out"]),
    }
    args.update(over)
    cmd = [sys.executable, "-m", "scripts.import_base_models", "--datasets", *DATASETS]
    for flag, value in args.items():
        if value is not None:
            cmd += [flag, value]
    return cmd


def _run(ws, **over):
    ws["pipeline"].write_text(PIPELINE, encoding="utf-8")
    return subprocess.run(_cmd(ws, **over), cwd=REPO_ROOT, capture_output=True, text=True)


# ------------------------------------------------------------------ 正常路径
def test_imports_only_models_and_leaves_relation_tables_empty(workspace):
    proc = _run(workspace)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    store = ModelStore(str(workspace["target"]))
    got = {r[0] for r in store.connection.execute("SELECT model_id FROM models")}
    assert got == {f"{d}__h1__{m}" for d in DATASETS for m in CANDIDATES}
    assert len(got) == len(DATASETS) * len(CANDIDATES)
    for table in RELATION_TABLES:
        count = store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert count == 0, f"{table} 不该被搬过来，实际 {count} 行"
    for model_id in sorted(got):
        row = store.get_model(model_id)
        assert row["lifecycle_stage"] == "active"
        assert Path(row["artifact_path"]).exists()
    store.close()

    report = json.loads((workspace["out"] / IMPORT_REPORT_FILENAME).read_text())
    assert report["base_horizon"] == 1
    assert report["imported_model_count"] == len(DATASETS) * len(CANDIDATES)
    assert report["all_artifacts_present"] is True
    assert sorted(report["models"]) == sorted(CANDIDATES)
    assert report["s1_history_starts"]["pjm"] == S1_START
    assert report["training_ranges"]["pjm"]["end"] == "2026-02-01 00:00:00"
    assert "sha256" not in json.dumps(report).lower()


def test_source_database_is_not_modified(workspace):
    before = workspace["source"].read_bytes()
    assert _run(workspace).returncode == 0
    assert workspace["source"].read_bytes() == before, "源库必须只读"


# --------------------------------------------------------------- 训练截止门控
@pytest.mark.parametrize("end", ["2026-03-01 00:00:00", "2026-03-05 00:00:00"])
def test_training_reaching_s1_fails_before_creating_the_target(workspace, end):
    """训练数据触到或越过 S1 时必须失败，且不留下目标库和报告。"""
    _write_train(workspace["features"], end=end)

    proc = _run(workspace)

    assert proc.returncode != 0
    assert "S1" in (proc.stdout + proc.stderr)
    assert not workspace["target"].exists(), "门控没过时不得创建目标库"
    assert not (workspace["out"] / IMPORT_REPORT_FILENAME).exists()


def test_gate_reads_the_csv_again_not_the_old_database(workspace):
    """判据来自重新读取的 train.csv，不是旧库里记的任何时间。"""
    starts = s1_history_starts(workspace["plan"], DATASETS)
    ranges = assert_training_ends_before_s1(workspace["features"], DATASETS, starts)
    assert ranges["pjm"]["end"] == "2026-02-01 00:00:00"
    assert ranges["pjm"]["path"].endswith("features/pjm/train.csv")

    _write_train(workspace["features"], end="2026-06-01 00:00:00")
    with pytest.raises(ImportError_, match="S1"):
        assert_training_ends_before_s1(workspace["features"], DATASETS, starts)


def test_gate_uses_explicit_b_training_cutoff_when_present(tmp_path):
    """窗口计划带显式 B training_cutoff 时，门控必须用它而不是 S1 历史起点。"""
    plan = tmp_path / "plan.json"
    _write_window_plan(plan)
    payload = json.loads(plan.read_text())
    for entry in payload["datasets"]:
        entry["training_cutoff"] = "2026-04-01 00:00:00"
    plan.write_text(json.dumps(payload), encoding="utf-8")

    features = tmp_path / "features"
    _write_train(features, end="2026-03-15 00:00:00")  # 晚于 S1、早于显式 cutoff

    starts = s1_history_starts(plan, DATASETS)
    assert starts["pjm"] == pd.Timestamp("2026-04-01 00:00:00")
    ranges = assert_training_ends_before_s1(features, DATASETS, starts)
    assert ranges["pjm"]["end"] == "2026-03-15 00:00:00"


# ------------------------------------------------------------------ 完整性
def test_missing_candidate_in_source_is_refused(workspace):
    source = _build_source(workspace["tmp"] / "partial", models=(CANDIDATES[0],))
    proc = _run(workspace, **{"--source": str(source)})
    assert proc.returncode != 0
    assert CANDIDATES[1] in (proc.stdout + proc.stderr)
    assert not workspace["target"].exists()


def test_missing_artifact_is_refused(workspace):
    source = _build_source(workspace["tmp"] / "noart", artifacts=False)
    with pytest.raises(ImportError_, match="产物不存在"):
        read_source_models(source, DATASETS, CANDIDATES)


def test_non_active_lifecycle_is_refused(workspace):
    source = _build_source(workspace["tmp"] / "shadow", lifecycle="shadow")
    with pytest.raises(ImportError_, match="lifecycle_stage"):
        read_source_models(source, DATASETS, CANDIDATES)


def test_source_with_undeclared_models_is_refused(workspace):
    """源库里有本次未声明的模型时不做部分导入。"""
    source = _build_source(workspace["tmp"] / "extra",
                           models=(*CANDIDATES, "xgboost_reg"))
    with pytest.raises(ImportError_, match="未声明|不做部分导入"):
        read_source_models(source, DATASETS, CANDIDATES)


def test_existing_target_is_never_overwritten(workspace):
    workspace["target"].parent.mkdir(parents=True, exist_ok=True)
    workspace["target"].write_bytes(b"existing")
    rows = read_source_models(workspace["source"], DATASETS, CANDIDATES)
    with pytest.raises(ImportError_, match="已存在"):
        write_target(workspace["target"], rows)
    assert workspace["target"].read_bytes() == b"existing"


def test_unparseable_training_timestamp_is_refused(workspace):
    """坏时间戳必须报错，不能被静默丢弃后让门控继续通过。"""
    path = workspace["features"] / "pjm" / "train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "timestamp"] = "不是时间"
    frame.to_csv(path, index=False)

    proc = _run(workspace)

    assert proc.returncode != 0
    assert "无法解析的时间戳" in (proc.stdout + proc.stderr)
    assert not workspace["target"].exists(), "坏输入下不得创建目标库"
