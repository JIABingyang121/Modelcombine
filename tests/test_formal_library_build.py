"""正式建库的两条硬约束：干净的新库，且 S/A 阶段看不到 T1—T3 的负荷值。

**为什么需要"干净的新库"**：往已有关系的库里再建一次库不会报错——`add_data_profile`
每次自增主键，`UNIQUE(scenario_id, data_profile_id, combination_id)` 因此永不触发，
旧关系会和新关系并存，在线匹配再把两批一起排序。这条必须在建库入口挡住。

**怎么证明 S/A 阶段没读 T1—T3 的负荷值**：把 T1—T3 目标区间的 `load` 写成非数字。
整表读取时 `pd.to_numeric` 会当场报错；只读到 A 窗口末尾则毫无影响。这比断言"某个变量
没被访问"更直接，也不依赖实现细节。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.storage.model_store import ModelStore
from tests.forecast_steps_fixtures import (
    DATASET,
    FIXTURE_CANDIDATES,
    REPO_ROOT,
    seed_models,
    write_dataset,
    write_frozen_window_plan,
)

ROWS = 1000
STEPS = 24
_ALL_METHODS = [
    "itransformer", "modelcombine", "mole", "random_forest",
    "stack_ensembles_reproduction", "time_moe", "xgboost",
]


def _poison_test_window_loads(raw_root: Path, plan_path: Path) -> pd.Timestamp:
    """把 A 窗口最后一个目标之后的 load 全部写成非数字，返回该边界。"""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    origins = {o["label"]: o for o in plan["datasets"][0]["origins"]}
    boundary = pd.Timestamp(origins["A"]["targets"][str(STEPS)]["last_target"])

    path = raw_root / DATASET / "load.csv"
    frame = pd.read_csv(path)
    stamps = pd.to_datetime(frame["timestamp"])
    frame["load"] = frame["load"].astype(object)
    frame.loc[stamps > boundary, "load"] = "T窗口真值不得在建库阶段被读取"
    frame.to_csv(path, index=False)
    assert (stamps > boundary).sum() > 0, "装置无效：边界之后没有任何行"
    return boundary


def _build(tmp_path: Path, *, db: Path, raw_root: Path, plan: Path, out: Path):
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES,
            "--raw-root", str(raw_root), "--window-plan", str(plan),
            "--out-root", str(out), "--database", str(db),
            "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


@pytest.fixture
def formal(tmp_path):
    raw_root = tmp_path / "raw"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    seed_models(db, tmp_path / "artifacts", frames["train"])
    plan = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)
    return {"db": db, "raw_root": raw_root, "plan": plan,
            "out": tmp_path / "out", "tmp": tmp_path}


# ------------------------------------------- S/A 阶段不得读取 T1—T3 的负荷值
def test_build_never_reads_test_window_loads(formal):
    boundary = _poison_test_window_loads(formal["raw_root"], formal["plan"])

    proc = _build(formal["tmp"], **{k: formal[k] for k in ("db", "raw_root", "plan", "out")})

    assert proc.returncode == 0, (
        "T1—T3 的 load 被读取了——整表 to_numeric 会在非数字上报错\n"
        + proc.stdout + proc.stderr
    )
    report = json.loads((formal["out"] / "model_library_report.json").read_text())
    assert len(report["tasks"]) == 3, "S1/S2/S3 各一条关系"
    assert boundary is not None


def test_freeze_reads_timestamps_only(formal):
    """冻结只需要数据覆盖范围；负荷值是非数字也不该影响它。

    直接测 `_dataset_definition`——它才是冻结里读原始序列的那一步。CLI 另有"恰好 7 方法 /
    3 数据集 / 3 长度"的正式范围强制，用它做这条断言反而测不到读取行为。
    """
    from scripts.freeze_final_experiment import _dataset_definition

    _poison_test_window_loads(formal["raw_root"], formal["plan"])
    plan = json.loads(formal["plan"].read_text(encoding="utf-8"))
    plan["datasets"][0]["fits"] = True

    entry = _dataset_definition(formal["raw_root"], plan, DATASET)

    expected_rows = len(pd.read_csv(formal["raw_root"] / DATASET / "load.csv"))
    assert entry["rows"] == expected_rows
    assert entry["training_rows"] > 0
    assert [w["label"] for w in entry["windows"]] == [
        "S1", "S2", "S3", "A", "T1", "T2", "T3"]


def test_formal_freeze_entry_enforces_full_scope(monkeypatch):
    """正式冻结的唯一入口必须先过范围强制，再进底层构造函数。

    这里 mock 掉底层构造，只验证包装层的拦截与放行——不需要把小装置膨胀成 3×3×7。
    """
    import scripts.freeze_final_experiment as freeze
    from scripts.freeze_final_experiment import FreezeError, build_formal_definition

    calls = []
    monkeypatch.setattr(freeze, "build_definition",
                        lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(freeze, "repo_state", lambda: ("f" * 40, []))
    full = dict(methods=list(_ALL_METHODS),
                datasets=["pjm_rto", "aemo_vic", "aemo_nsw"],
                forecast_steps=[24, 168, 720])

    # 少一个方法 / 少一个数据集 / 少一个长度，都不得进入底层构造
    for over, needle in [
        ({"methods": _ALL_METHODS[:-1]}, "全部已注册方法"),
        ({"datasets": ["pjm_rto", "aemo_vic"]}, "数据集必须恰好"),
        ({"forecast_steps": [24, 168]}, "预测长度必须恰好"),
    ]:
        with pytest.raises(FreezeError, match=needle):
            build_formal_definition(**{**full, **over})
    assert calls == [], "范围不完整时底层构造函数不得被调用"

    assert build_formal_definition(**full) == {"ok": True}
    assert len(calls) == 1, "范围完整时才进入底层构造函数"


def test_formal_freeze_refuses_a_dirty_worktree(monkeypatch):
    """脏代码不得生成定义——否则定义会记着一个干净 HEAD，运行侧再也证明不了来源。"""
    import scripts.freeze_final_experiment as freeze
    from scripts.freeze_final_experiment import FreezeError, build_formal_definition

    calls = []
    monkeypatch.setattr(freeze, "build_definition",
                        lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(freeze, "repo_state",
                        lambda: ("c" * 40, [" M scripts/final_comparison.py"]))

    with pytest.raises(FreezeError, match="未提交修改"):
        build_formal_definition(methods=list(_ALL_METHODS),
                                datasets=["pjm_rto", "aemo_vic", "aemo_nsw"],
                                forecast_steps=[24, 168, 720])
    assert calls == [], "脏工作树时底层构造函数不得被调用"


def test_formal_freeze_writes_the_actual_head(monkeypatch):
    """干净工作树：写进定义的必须是实际 HEAD。"""
    import scripts.freeze_final_experiment as freeze
    from scripts.freeze_final_experiment import build_formal_definition

    calls = []
    monkeypatch.setattr(freeze, "build_definition",
                        lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(freeze, "repo_state", lambda: ("d" * 40, []))

    build_formal_definition(methods=list(_ALL_METHODS),
                            datasets=["pjm_rto", "aemo_vic", "aemo_nsw"],
                            forecast_steps=[24, 168, 720])

    assert calls[0]["repo_commit"] == "d" * 40


def test_formal_freeze_refuses_a_caller_supplied_commit(monkeypatch):
    """调用方不得自带 repo_commit——那等于伪造定义来源。"""
    import scripts.freeze_final_experiment as freeze
    from scripts.freeze_final_experiment import FreezeError, build_formal_definition

    calls = []
    monkeypatch.setattr(freeze, "build_definition",
                        lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(freeze, "repo_state", lambda: ("d" * 40, []))

    with pytest.raises(FreezeError, match="不接受调用方提供的 repo_commit"):
        build_formal_definition(methods=list(_ALL_METHODS),
                                datasets=["pjm_rto", "aemo_vic", "aemo_nsw"],
                                forecast_steps=[24, 168, 720],
                                repo_commit="e" * 40)
    assert calls == []


def test_low_level_builder_is_not_the_formal_entry():
    """底层构造函数不强制正式范围——这是它可测的前提，也是必须走包装层的理由。"""
    import inspect

    from scripts.freeze_final_experiment import build_definition, build_formal_definition

    assert "不是正式冻结入口" in (build_definition.__doc__ or "")
    assert "唯一入口" in (build_formal_definition.__doc__ or "")
    # 包装层不得提供任何绕过参数
    params = set(inspect.signature(build_formal_definition).parameters)
    assert not {"bypass", "skip_scope_check", "enforce_formal_scope"} & params


# ------------------------------------------------------ 正式建库要求干净新库
def test_formal_build_refuses_a_library_that_already_has_relations(formal):
    first = _build(formal["tmp"], **{k: formal[k] for k in ("db", "raw_root", "plan", "out")})
    assert first.returncode == 0, first.stdout + first.stderr
    store = ModelStore(str(formal["db"]))
    before = store.connection.execute(
        "SELECT COUNT(*) FROM scenario_data_combinations").fetchone()[0]
    store.close()
    assert before == 3

    again = _build(formal["tmp"], db=formal["db"], raw_root=formal["raw_root"],
                   plan=formal["plan"], out=formal["tmp"] / "out2")

    assert again.returncode != 0, "对同一个库重建会把新关系追加到旧关系旁边"
    assert "干净的新库" in (again.stdout + again.stderr)
    store = ModelStore(str(formal["db"]))
    after = store.connection.execute(
        "SELECT COUNT(*) FROM scenario_data_combinations").fetchone()[0]
    store.close()
    assert after == before, "被拒绝后不得留下任何新关系"


def test_formal_build_refuses_when_models_do_not_match_declared_candidates(formal):
    proc = _build(formal["tmp"], db=formal["db"], raw_root=formal["raw_root"],
                  plan=formal["plan"], out=formal["out"])
    assert proc.returncode == 0

    # 换一个空库，只登记一部分候选
    partial = formal["tmp"] / "partial.sqlite3"
    frames = write_dataset(formal["tmp"] / "splits2", rows=ROWS)
    seed_models(partial, formal["tmp"] / "artifacts2", frames["train"])
    store = ModelStore(str(partial))
    with store.connection:
        store.connection.execute(
            "DELETE FROM models WHERE model_id = ?", (f"{DATASET}__h1__lgbm_reg",))
    store.close()

    again = _build(formal["tmp"], db=partial, raw_root=formal["raw_root"],
                   plan=formal["plan"], out=formal["tmp"] / "out3")

    assert again.returncode != 0
    assert "lgbm_reg" in (again.stdout + again.stderr)
