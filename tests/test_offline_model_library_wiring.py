"""离线不限成员数量的最佳组合构建（按完整轨迹，方案 §3.3）。

通过真实 scripts/train_combinations_kg.py --model-library 入口运行：
- 所有非空子集都被枚举，不写死二/三模型上限；
- test 切分（标签与候选预测分布）都不影响组合选择；
- 声明的候选未登记 / 产物丢失 -> 建库不完整，整体失败；
- 已登记但不符合历史数据契约的候选 -> 记录为无资格，不算失败；
- 组合器保存后不可重放时，任何数据库写入都不发生；
- 原始数据缺失的任务直接失败，不静默产出报告。

用户预测长度契约本身在 tests/test_forecast_steps_contract.py。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.combination_utils import load_predictions_safe
from src.models.artifacts import load_artifact
from src.storage.model_store import ModelStore
from tests.forecast_steps_fixtures import (
    DATASET,
    FIXTURE_CANDIDATES,
    REPO_ROOT,
    fit_weekly,
    register_models,
    run_build,
    seed_models,
    task_of,
    write_dataset,
    write_frozen_window_plan,
)

ROWS = 1000
STEPS = 24


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


def _prepare(tmp_path: Path):
    """真实原始数据 + 真实登记的 h=1 候选模型产物。"""
    raw_root = tmp_path / "features"
    artifacts = tmp_path / "artifacts"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(raw_root, rows=ROWS)
    seed_models(db, artifacts, frames["train"])
    return raw_root, db, frames


def _run_library(tmp_path: Path, raw_root: Path, db: Path, out_root: Path):
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES,
            "--raw-root", str(raw_root), "--out-root", str(out_root),
            "--database", str(db), "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_window_plan_builds_three_relations_against_one_shared_audit_window(tmp_path):
    """Stage 1：S1—S3 分别建库，共用 A；T1—T3 尚不进入建库。"""
    raw_root = tmp_path / "raw"
    artifacts = tmp_path / "artifacts"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    seed_models(db, artifacts, frames["train"])
    window_plan = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)
    out_root = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES,
            "--raw-root", str(raw_root), "--window-plan", str(window_plan),
            "--out-root", str(out_root), "--database", str(db),
            "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    tasks = json.loads((out_root / "model_library_report.json").read_text())["tasks"]
    assert [task["scenario_sample"] for task in tasks] == ["S1", "S2", "S3"]
    assert {task["audit_window"] for task in tasks} == {"A"}
    assert len({task["test_origin"] for task in tasks}) == 1


def test_all_non_empty_subsets_of_safe_models_are_enumerated(tmp_path):
    """§3.3.3：继续枚举所有非空子集，不限制二模型或三模型。"""
    result = run_build(tmp_path, [STEPS], rows=ROWS)
    task = task_of(result["report"], STEPS)

    requested = {tuple(entry["members"]) for entry in task["enumerated_combinations"]}
    safe = task["safe_models"]
    assert len(safe) >= 2, "装置至少要留下两个安全候选，否则枚举无从谈起"
    # 去重键是"实际生效成员"，被近零权重清理后可能与请求集合合并，
    # 因此枚举结果数不超过 2^k-1，且每个结果都必须是安全候选的子集
    assert 0 < len(requested) <= 2 ** len(safe) - 1
    for members in requested:
        assert set(members) <= set(safe)
    assert any(len(members) >= 2 for members in requested)


def test_selection_never_receives_test_data(tmp_path):
    """§3.3.6 的结构性保证：组合枚举这一步拿不到任何 test 数据。

    这是唯一不依赖数值巧合的断言。Protocol A 会用 val/test 两侧的**预测分布**估
    PSI（src/eval/kg/drift.py:50），Protocol B 再据此做稳定性过滤与 interaction
    门控——只要真实 test 预测进入枚举调用，即使 test 标签完全没参与拟合，拟合出的
    权重也会变（已实测：只平移 test 候选预测，val 权重从 0.5738 变到 0.6108）。
    """
    import scripts.train_combinations_kg as tck

    raw_root, db, _frames = _prepare(tmp_path)
    store = ModelStore(str(db))
    real_eval = tck.evaluate_fixed_protocol_b_combination
    seen = []

    def _recording_eval(df_val, df_test, raw_val, raw_test, **kwargs):
        seen.append(bool(df_val.equals(df_test) and raw_val.equals(raw_test)))
        return real_eval(df_val, df_test, raw_val, raw_test, **kwargs)

    tck.evaluate_fixed_protocol_b_combination = _recording_eval
    try:
        tck._build_library_task(
            store,
            dataset=DATASET,
            forecast_steps=STEPS,
            model_types=FIXTURE_CANDIDATES,
            raw_root=raw_root,
            artifact_dir=tmp_path / "combo_artifacts",
            filter_threshold=2.0,
        )
    finally:
        tck.evaluate_fixed_protocol_b_combination = real_eval
        store.close()

    assert seen, "组合枚举一次都没跑，断言无意义"
    assert all(seen), f"有 {seen.count(False)}/{len(seen)} 次枚举调用拿到了 test 数据"


def test_replacing_the_test_split_does_not_change_selection(tmp_path):
    """把整个 test 切分换成 val 的副本：validation 输入一字节未变，
    因此成员、权重、validation MAE 和 val 轨迹都必须完全一致。

    这个改法故意让 test 侧的 PSI 从"与 val 明显不同"变成"完全相同"，不依赖某个
    具体扰动幅度是否恰好跨过漂移阈值。
    """
    baseline = run_build(tmp_path / "a", [STEPS], rows=ROWS)
    task_a = task_of(baseline["report"], STEPS)

    raw_root, db, _frames = _prepare(tmp_path / "b")
    shutil.copyfile(raw_root / DATASET / "val.csv", raw_root / DATASET / "test.csv")

    proc = _run_library(tmp_path / "b", raw_root, db, tmp_path / "b" / "out")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    task_b = task_of(
        json.loads((tmp_path / "b" / "out" / "model_library_report.json").read_text()), STEPS
    )

    assert task_a["effective_members"] == task_b["effective_members"]
    assert task_a["linear_weights"] == pytest.approx(task_b["linear_weights"], abs=1e-12)
    assert task_a["validation_mae"] == pytest.approx(task_b["validation_mae"], abs=1e-12)
    np.testing.assert_allclose(
        np.asarray(task_a["val_trajectory"], dtype=float),
        np.asarray(task_b["val_trajectory"], dtype=float),
        rtol=0, atol=1e-12,
    )
    # 换掉的 test 确实是另一批数据：冻结后评价的 test 指标必须不同
    assert task_a["test_mae"] != pytest.approx(task_b["test_mae"], abs=1e-6)


def test_registered_model_with_missing_artifact_fails_the_whole_build(tmp_path):
    """已登记但产物文件丢失 = 模型库损坏，不允许静默缩小候选池。"""
    raw_root, db, _frames = _prepare(tmp_path)
    store = ModelStore(str(db))
    victim = store.get_model(f"{DATASET}__h1__catboost_reg")
    store.close()
    Path(victim["artifact_path"]).unlink()

    proc = _run_library(tmp_path, raw_root, db, tmp_path / "out")

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "catboost_reg" in combined and "产物文件" in combined
    assert not (tmp_path / "out" / "model_library_report.json").exists()


def test_build_rejects_non_replayable_combination(tmp_path):
    """保存后重放对不上 1e-8 时，scenario/组合/关系一行都不写。"""
    import scripts.train_combinations_kg as tck

    raw_root, db, _frames = _prepare(tmp_path)
    store = ModelStore(str(db))
    real_load = tck.load_artifact

    def _perturbing_load(path):
        artifact = real_load(path)
        if str(path).endswith("__combo.pkl"):
            base_predict = artifact.predict
            artifact.predict = lambda *a, **k: np.asarray(base_predict(*a, **k), dtype=float) + 1.0
        return artifact

    tck.load_artifact = _perturbing_load
    try:
        with pytest.raises(RuntimeError, match="重放误差"):
            tck._build_library_task(
                store,
                dataset=DATASET,
                forecast_steps=STEPS,
                model_types=FIXTURE_CANDIDATES,
                raw_root=raw_root,
                artifact_dir=tmp_path / "combo_artifacts",
                filter_threshold=2.0,
            )
    finally:
        tck.load_artifact = real_load

    for table in ("scenarios", "data_profiles", "combinations", "scenario_data_combinations"):
        assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    store.close()


def test_build_fails_loudly_when_raw_data_is_missing(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out_root = tmp_path / "out"
    db = tmp_path / "lib.sqlite3"
    store = ModelStore(str(db))
    store.create_schema()
    store.close()

    proc = _run_library(tmp_path, tmp_path / "features", db, out_root)

    assert proc.returncode != 0
    assert not (out_root / "model_library_report.json").exists()


def test_build_fails_when_split_is_too_short_for_signature_window(tmp_path):
    """§6.1：预测起点前必须有完整的 720 小时 signature 窗口，不压缩窗口继续跑。"""
    raw_root, db, _frames = _prepare(tmp_path)
    short = pd.read_csv(raw_root / DATASET / "val.csv").head(600)
    short.to_csv(raw_root / DATASET / "val.csv", index=False)

    proc = _run_library(tmp_path, raw_root, db, tmp_path / "out")

    assert proc.returncode != 0
    assert "signature" in (proc.stdout + proc.stderr)


def test_declared_candidate_not_registered_fails_the_whole_build(tmp_path):
    """声明的候选没登记 = 建库不完整，整体失败，不产出报告。

    候选集合与 configs/pipeline.yaml 的 models: 段同源，正是 train_baselines
    会登记的那一批；少一个就说明训练环节没跑全，不能拿缩水的池子出正式数字。
    """
    raw_root, db, _frames = _prepare(tmp_path)

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            # prophet 在装置里没登记
            "--candidates", *FIXTURE_CANDIDATES, "prophet",
            "--raw-root", str(raw_root), "--out-root", str(tmp_path / "out"),
            "--database", str(db), "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "prophet" in combined and "未登记" in combined
    assert not (tmp_path / "out" / "model_library_report.json").exists()


def test_registered_but_contract_ineligible_candidate_is_recorded_not_fatal(tmp_path):
    """已登记、但在历史数据契约下产不出轨迹的候选，是可记录的"无资格"，不是失败。

    这条和上一条一起构成候选完整性规则的两侧：缺产物 = 建库不完整；
    有产物但不符合输入契约 = 记录原因后排除。
    """
    raw_root, db, frames = _prepare(tmp_path)
    # arima 有产物，但其 predict 只从训练序列末尾外推，读不到用户历史
    register_models(db, tmp_path / "artifacts", [
        ("arima", fit_weekly(frames["train"]), ["hour", "lag_24"]),
    ])

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES, "arima",
            "--raw-root", str(raw_root), "--out-root", str(tmp_path / "out"),
            "--database", str(db), "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    task = task_of(
        json.loads((tmp_path / "out" / "model_library_report.json").read_text()), STEPS
    )
    reasons = {entry["model_type"]: entry["reason"] for entry in task["skipped_candidates"]}
    assert "arima" in reasons and "无法消费用户提交的最新历史" in reasons["arima"]
    assert "arima" not in {m.split("__")[-1] for m in task["effective_members"]}
    # 声明过就要在报告里留痕，便于验收文件核对候选完整性
    assert set(task["declared_candidates"]) == set(FIXTURE_CANDIDATES) | {"arima"}


def test_single_member_relation_uses_the_model_as_is(tmp_path):
    """最终关系只剩一个模型时，就是"用这个模型"，不是"这个模型的加权组合"。

    让 Ridge 自由拟合单列系数得到的是一个一般不等于 1 的缩放（关掉本修正后本装置
    实测 1.0087，另一个装置上是 0.9993，两个方向都出现过），等于给单模型关系加了一个
    无理由的系统性偏置。装置只声明两个候选，其中 xgboost_reg 需要未来 temp、按输入
    契约无资格，于是只剩一个合格候选，必定产出单成员关系——不靠运气命中该分支。
    """
    result = run_build(
        tmp_path, [STEPS], rows=ROWS, candidates=["catboost_reg", "xgboost_reg"]
    )
    task = task_of(result["report"], STEPS)

    assert task["effective_members"] == [f"{DATASET}__h1__catboost_reg"]
    assert task["linear_weights"] == [1.0]
    assert task["has_interaction"] is False

    # 记录的 validation MAE 就是该成员自己的轨迹 MAE，不是被收缩后的组合 MAE
    candidate_mae = task["candidate_validation_mae"]["catboost_reg"]["trajectory_mae"]
    assert task["validation_mae"] == pytest.approx(candidate_mae, abs=1e-12)

    store = ModelStore(str(result["db"]))
    combo = store.get_combination(task["combination_id"])
    store.close()
    assert [m["weight"] for m in combo["members"]] == [1.0]

    predictor = load_artifact(combo["artifact_path"])
    assert predictor.linear_weights == [1.0]
    assert predictor.interaction is None
    assert predictor.required_feature_names == []
    # 原样使用：给定成员轨迹，组合输出就等于它本身
    probe = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        predictor.predict({predictor.member_ids[0]: probe}), probe, rtol=0, atol=0
    )


def test_report_names_samples_and_horizons_without_conflating_them(tmp_path):
    """S1/S2/S3 是历史数据样例；H1/H2/H3 只表示预测长度（24/168/720）。

    数据库真正的字段仍是 forecast_steps，不新增重复的 H 字段。
    """
    from scripts.train_combinations_kg import FORECAST_HORIZON_LABELS

    assert FORECAST_HORIZON_LABELS == {24: "H1", 168: "H2", 720: "H3"}

    raw_root = tmp_path / "raw"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    seed_models(db, tmp_path / "artifacts", frames["train"])
    window_plan = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)
    out_root = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES,
            "--raw-root", str(raw_root), "--window-plan", str(window_plan),
            "--out-root", str(out_root), "--database", str(db),
            "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    tasks = json.loads((out_root / "model_library_report.json").read_text())["tasks"]

    # 三段历史数据样例叫 S1/S2/S3，不再借用 H1/H2/H3
    assert [t["scenario_sample"] for t in tasks] == ["S1", "S2", "S3"]
    # 预测长度别名与 forecast_steps 一一对应
    assert {t["forecast_horizon"] for t in tasks} == {FORECAST_HORIZON_LABELS[STEPS]}
    assert {t["forecast_steps"] for t in tasks} == {STEPS}

    # 数据库里没有多出一个 H 字段，forecast_steps 仍是唯一真源
    store = ModelStore(str(db))
    columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(scenarios)")}
    store.close()
    assert "forecast_steps" in columns
    assert not any(c in columns for c in ("forecast_horizon", "horizon_label", "h_label"))
