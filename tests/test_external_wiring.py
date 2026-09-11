"""外部方法经统一入口跑 T1—T3 的真实接线。

已有的 test_external_adapters 只覆盖输入辅助函数、命令构造和输出解析；本文件覆盖
**接线**：训练文件与当前查询历史必须分开、训练按 dataset×forecast_steps×seed 只做一次
并在 T1—T3 复用、请求的种子必须真正传给官方方法。

subprocess 调用本身仍不在本机测（没有官方仓库也没有 GPU）；这里把官方调用替换成记录
调用参数的探针，验证的是**我们喂给官方什么**，不是官方内部行为。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import final_comparison
from scripts.final_comparison import FinalComparisonError, run
from tests.forecast_steps_fixtures import (
    DATASET,
    write_dataset,
    write_frozen_window_plan,
)

STEPS = 24
ROWS = 1800
WINDOWS = ("T1", "T2", "T3")


@pytest.fixture
def wiring(tmp_path, monkeypatch):
    """真实窗口计划 + 记录式官方调用探针。"""
    raw_root = tmp_path / "raw"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    plan = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)

    repo = tmp_path / "external_repo"
    (repo / "dataset").mkdir(parents=True)
    external = {
        method: {
            "repo": str(repo), "python": "/nonexistent/python",
            "checkpoints": str(tmp_path / "ckpt"), "snapshot": str(tmp_path / "snap"),
            "hyperparameters": {
                "seq_len": 96, "label_len": 48, "d_model": 512, "n_heads": 8,
                "e_layers": 3, "d_layers": 1, "d_ff": 512, "factor": 1,
                "dropout": 0.1, "train_epochs": 10, "batch_size": 32,
                "patience": 3, "learning_rate": 0.0001, "t_dim": 4, "des": "final",
            "split_policy": "adaptive_val_at_least_one_full_batch",
            },
        }
        for method in ("itransformer", "mole", "time_moe")
    }

    calls = []

    ckpt_root = tmp_path / "ckpt"

    def _record(command, *, cwd, method):
        command = [str(c) for c in command]
        calls.append({"method": method, "command": command})
        if method == "mole" and "--is_training" in command:
            # 官方训练写出 checkpoint。目录名由官方 setting 决定，含 seq_len 与 des——
            # 这正是 model_id 区分不开冒烟与正式两套产物的原因。
            folder = ckpt_root / "_".join([
                command[command.index("--model_id") + 1], "MoLE_DLinear", "custom", "ftS",
                "sl" + command[command.index("--seq_len") + 1],
                "pl" + command[command.index("--pred_len") + 1],
                command[command.index("--des") + 1], "0",
            ])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "checkpoint.pth").write_text("checkpoint", encoding="utf-8")

    reads = []

    def _fake_output(path, method, forecast_steps):
        reads.append(str(path))
        return np.full(forecast_steps, 1000.0)

    monkeypatch.setattr(final_comparison, "run_official", _record)
    monkeypatch.setattr(final_comparison, "read_official_output", _fake_output)
    return {
        "raw_root": raw_root, "plan": plan, "external": external,
        "calls": calls, "reads": reads, "repo": repo, "ckpt_root": ckpt_root,
        "record": _record,
    }


def _run(wiring, method, *, seeds=(42,), windows=WINDOWS):
    return run(
        methods=[method], datasets=[DATASET], windows=list(windows),
        forecast_steps=[STEPS], seeds=list(seeds),
        raw_root=wiring["raw_root"], window_plan=wiring["plan"], database=None,
        external=wiring["external"],
    )


def _written(repo: Path, pattern: str):
    return sorted((repo / "dataset").glob(pattern))


# ------------------------------------------------- P0-1 训练文件 vs 查询历史
def test_training_file_is_frozen_and_query_history_ends_at_each_forecast_origin(wiring):
    """训练文件按 training_cutoff 冻结；每个 T 窗口的查询历史结束于该窗口预测起点。

    这正是旧实现坏掉的地方：它用 training_cutoff 去截当前窗口历史，导致 T1 只剩 1 行、
    T2/T3 直接为空。
    """
    frame = _run(wiring, "mole")
    assert len(frame) == len(WINDOWS) * STEPS

    plan = json.loads(wiring["plan"].read_text())
    origins = {o["label"]: o for o in plan["datasets"][0]["origins"]}
    cutoff = pd.Timestamp(origins["T1"]["history_start"])

    training_files = _written(wiring["repo"], "mc_split_*/train.csv")
    assert len(training_files) == 1, "训练文件应按 dataset×forecast_steps×seed 只写一份"
    train = pd.read_csv(training_files[0])
    assert list(train.columns) == ["date", "load"]
    assert pd.to_datetime(train["date"]).max() < cutoff
    assert len(train) > 100, "训练文件不能被截成一行"

    prediction_files = _written(wiring["repo"], "mc_pred_*.csv")
    assert len(prediction_files) == len(WINDOWS)
    for label in WINDOWS:
        path = next(p for p in prediction_files if f"_{label}_" in p.name)
        history = pd.read_csv(path)
        assert not history.empty, f"{label} 的查询历史不能为空"
        # 查询历史必须结束于该窗口的预测起点，而不是训练截止点
        assert pd.to_datetime(history["date"]).max() == pd.Timestamp(
            origins[label]["forecast_origin"]
        )
        assert len(history) == 720


def test_training_happens_once_and_is_reused_across_all_test_windows(wiring):
    """iTransformer / MoLE 必须只训练一次，T1—T3 复用同一个 checkpoint。"""
    _run(wiring, "mole")

    trains = [c for c in wiring["calls"] if "--is_training" in c["command"]]
    predicts = [c for c in wiring["calls"] if "--is_training" not in c["command"]]
    assert len(trains) == 1, f"应只训练一次，实际 {len(trains)} 次"
    assert len(predicts) == len(WINDOWS)


# ------------------------------------------------------------ P0-2 真实种子
def test_requested_seed_reaches_the_official_command(wiring):
    """请求 43 就必须真的把 43 传给官方，而不是永远用 42。"""
    _run(wiring, "mole", seeds=(43,), windows=("T1",))

    train = next(c for c in wiring["calls"] if "--is_training" in c["command"])
    assert train["command"][train["command"].index("--seed") + 1] == "43"
    predict = next(c for c in wiring["calls"] if "--is_training" not in c["command"])
    assert "43" in predict["command"], "预测调用也必须收到请求的种子"


def test_different_seeds_do_not_share_artifact_paths(wiring):
    """不同种子的训练文件与产物路径必须区分，否则后一个种子会覆盖前一个。"""
    _run(wiring, "mole", seeds=(42, 43), windows=("T1",))

    training_files = _written(wiring["repo"], "mc_split_*/train.csv")
    assert len(training_files) == 2
    assert len({p.parent.name for p in training_files}) == 2
    model_ids = [
        c["command"][c["command"].index("--model_id") + 1]
        for c in wiring["calls"] if "--model_id" in c["command"]
    ]
    assert len(set(model_ids)) == len(model_ids), f"model_id 必须含种子: {model_ids}"


def test_time_moe_uses_the_current_window_history_not_the_training_cutoff(wiring):
    """Time-MoE 不做任务训练，应直接用当前窗口的 720 小时历史。"""
    _run(wiring, "time_moe", windows=("T2",))

    assert not _written(wiring["repo"], "mc_train_*.csv"), "Time-MoE 不需要训练文件"
    prediction_files = _written(wiring["repo"], "mc_pred_*.csv")
    assert len(prediction_files) == 1
    plan = json.loads(wiring["plan"].read_text())
    origins = {o["label"]: o for o in plan["datasets"][0]["origins"]}
    history = pd.read_csv(prediction_files[0])
    assert pd.to_datetime(history["date"]).max() == pd.Timestamp(
        origins["T2"]["forecast_origin"]
    )


# --------------------------------------------------- P0-3 正式超参数不得缺省
def test_formal_hyperparameters_must_come_from_config(wiring):
    """探针超参数（1 epoch 小模型）不得作为正式默认值悄悄生效。"""
    wiring["external"]["mole"].pop("hyperparameters")
    with pytest.raises(FinalComparisonError, match="hyperparameters"):
        _run(wiring, "mole", windows=("T1",))


def test_configured_hyperparameters_are_passed_through(wiring):
    _run(wiring, "itransformer", windows=("T1",))

    command = wiring["calls"][0]["command"]
    assert command[command.index("--train_epochs") + 1] == "10"
    assert command[command.index("--d_model") + 1] == "512"
    assert command[command.index("--e_layers") + 1] == "3"
    assert command[command.index("--des") + 1] == "final"


# ------------------------------------------- iTransformer 查询必须真的执行 predict
def test_itransformer_query_runs_official_predict_per_window(wiring):
    """官方 run.py 的 ``--is_training 0`` 分支只调 ``exp.test()``，且忽略 ``--do_predict``。

    只有 ``exp.predict()`` 写 ``real_prediction.npy``。旧接线因此在 T1—T3 都读到训练阶段
    留下的同一份旧文件。查询必须走官方 ``Exp.predict``，并写到窗口专属产物路径。
    """
    _run(wiring, "itransformer")

    # 训练与查询都走 python -c <片段>，按片段区分
    trains = [c for c in wiring["calls"]
              if c["command"][2] == final_comparison.ITRANSFORMER_TRAIN_SNIPPET]
    queries = [c for c in wiring["calls"]
               if c["command"][2] == final_comparison.ITRANSFORMER_PREDICT_SNIPPET]
    assert len(trains) == 1, f"应只训练一次，实际 {len(trains)} 次"
    train = trains[0]["command"]
    assert train[train.index("--is_training") + 1] == "1"
    # 训练阶段不再产出 real_prediction.npy——它正是被误读成查询结果的那份旧文件
    assert "--do_predict" not in train

    assert len(queries) == len(WINDOWS)
    outputs = []
    for call in queries:
        command = call["command"]
        assert command[1] == "-c"
        assert command[2] == final_comparison.ITRANSFORMER_PREDICT_SNIPPET
        outputs.append(command[3])
        # 查询命令必须带 --do_predict，并复用训练时的 model_id（即同一个 checkpoint）
        assert "--do_predict" in command
        assert command[command.index("--model_id") + 1] == train[train.index("--model_id") + 1]
    assert len(set(outputs)) == len(WINDOWS), f"三个窗口写到了同一路径: {outputs}"
    assert wiring["reads"] == outputs, "读回的必须正是本窗口刚写出的产物，不能靠 glob 捡旧文件"


def test_stale_training_file_from_a_smoke_run_is_not_reused(wiring):
    """冒烟阶段留下的同名训练文件，正式实验必须覆盖重写而不是直接复用。"""
    stale = (
        wiring["repo"] / "dataset" /
        f"mc_split_{DATASET}_h{STEPS}_s42" / "train.csv"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("date,load\n2000-01-01 00:00:00,1.0\n", encoding="utf-8")

    _run(wiring, "mole", windows=("T1",))

    train = pd.read_csv(stale)
    assert len(train) > 100, "训练文件必须按当前 raw 与 training_cutoff 重写"
    assert pd.to_datetime(train["date"]).min().year > 2000


# ------------------------------------------------- 冻结定义是外部方法的唯一口径
def test_frozen_external_values_are_used_and_conflicts_fail():
    frozen = {
        "itransformer": {
            "commit": "c2426e68ca13f74aaec08045c5c724d8ad328124",
            "hyperparameters": {"d_model": 512, "seq_len": 96},
        }
    }
    config = {"itransformer": {"repo": "/r", "python": "/p", "checkpoints": "/c"}}

    merged = final_comparison.apply_frozen_external(config, frozen)
    assert merged["itransformer"]["hyperparameters"] == {"d_model": 512, "seq_len": 96}
    assert merged["itransformer"]["commit"] == frozen["itransformer"]["commit"]
    assert merged["itransformer"]["repo"] == "/r"

    config["itransformer"]["hyperparameters"] = {"d_model": 64, "seq_len": 96}
    with pytest.raises(FinalComparisonError, match="冻结"):
        final_comparison.apply_frozen_external(config, frozen)


# ------------------------------------- MoLE 必须加载本次训练产出的那个 checkpoint
def test_mole_loads_the_checkpoint_this_run_trained_not_a_smoke_leftover(wiring):
    """官方 checkpoint 目录名含 seq_len 与 des，model_id 区分不开冒烟与正式两套产物。

    这里冒烟留下 ``sl336…probe``、正式训练产出 ``sl96…final``；按名字排序 ``sl336`` 在前，
    旧实现会静默加载探针小模型，正式结果被污染却不会报错。
    """
    stale = wiring["ckpt_root"] / (
        f"{DATASET}_h{STEPS}_s42_MoLE_DLinear_custom_ftS_sl336_pl{STEPS}_probe_0"
    )
    stale.mkdir(parents=True)
    (stale / "checkpoint.pth").write_text("probe", encoding="utf-8")
    os.utime(stale / "checkpoint.pth", (0, 0))

    _run(wiring, "mole", windows=WINDOWS)

    fresh = wiring["ckpt_root"] / (
        f"{DATASET}_h{STEPS}_s42_MoLE_DLinear_custom_ftS_sl96_pl{STEPS}_final_0"
    )
    predicts = [c for c in wiring["calls"] if "--is_training" not in c["command"]]
    assert len(predicts) == len(WINDOWS)
    for call in predicts:
        assert str(fresh / "checkpoint.pth") in call["command"]
        assert str(stale / "checkpoint.pth") not in call["command"]


def test_mole_detects_an_overwritten_checkpoint_at_the_same_path(wiring):
    """同一套超参数重跑时官方会原地覆盖 checkpoint.pth，也必须认定为本次产物。

    判据不能是"晚于训练开始时刻"：文件 mtime 用内核粗粒度时钟，time.time() 用细粒度时钟，
    刚写出的文件可能显示得比训练开始还早。
    """
    same = wiring["ckpt_root"] / (
        f"{DATASET}_h{STEPS}_s42_MoLE_DLinear_custom_ftS_sl96_pl{STEPS}_final_0"
    )
    same.mkdir(parents=True)
    (same / "checkpoint.pth").write_text("previous", encoding="utf-8")
    os.utime(same / "checkpoint.pth", (0, 0))

    _run(wiring, "mole", windows=("T1",))

    predict = next(c for c in wiring["calls"] if "--is_training" not in c["command"])
    assert str(same / "checkpoint.pth") in predict["command"]
    assert (same / "checkpoint.pth").read_text() == "checkpoint"


def test_mole_fails_loudly_when_training_produces_no_checkpoint(wiring, monkeypatch):
    """官方训练没留下新 checkpoint 时必须报错，不能回头去捡任何旧文件。"""
    stale = wiring["ckpt_root"] / (
        f"{DATASET}_h{STEPS}_s42_MoLE_DLinear_custom_ftS_sl96_pl{STEPS}_probe_0"
    )
    stale.mkdir(parents=True)
    (stale / "checkpoint.pth").write_text("probe", encoding="utf-8")
    os.utime(stale / "checkpoint.pth", (0, 0))
    monkeypatch.setattr(
        final_comparison, "run_official",
        lambda command, *, cwd, method: wiring["calls"].append(
            {"method": method, "command": [str(c) for c in command]}
        ),
    )

    with pytest.raises(FinalComparisonError, match="checkpoint"):
        _run(wiring, "mole", windows=("T1",))


# ----------------------------------------------- 种子网格必须在训练开始前就被拦住
def test_incompatible_seed_grids_are_rejected_before_any_training():
    definition = {"seeds": {
        "itransformer": [2023], "modelcombine": [42], "mole": [42, 43, 44],
    }}

    # 一种子方法可以同批：iTransformer 冻结 2023、Modelcombine 冻结 42，数量相同
    final_comparison.assert_seed_grid(["itransformer", "modelcombine"], [42], definition)

    # MoLE 要三种子，混进来必须直接报错，而不是让确定性方法白跑三遍
    with pytest.raises(FinalComparisonError, match="分批"):
        final_comparison.assert_seed_grid(
            ["itransformer", "mole"], [42, 43, 44], definition
        )
    with pytest.raises(FinalComparisonError, match="种子"):
        final_comparison.assert_seed_grid(["mole"], [42], definition)
    with pytest.raises(FinalComparisonError, match="种子"):
        final_comparison.assert_seed_grid(["modelcombine"], [43], definition)
    with pytest.raises(FinalComparisonError, match="冻结定义"):
        final_comparison.assert_seed_grid(["stack_ensembles"], [42], definition)


# ------------------------------------------- 冻结的官方版本必须与实际代码/权重对上
def _git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "a.txt").write_text("x", encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=path, check=True, env=env)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_external_versions_are_checked_against_the_actual_repo_and_snapshot(tmp_path):
    head = _git_repo(tmp_path / "mole_repo")
    revision = "3f2c1ab9de4057118cbb2d5f8a6e0c4d91b7a2e0"
    snapshot = tmp_path / "hf" / "models--Maple728--TimeMoE-50M" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    external = {
        "mole": {
            "repo": str(tmp_path / "mole_repo"), "python": "/p",
            "checkpoints": "/c", "commit": head,
        },
        "time_moe": {
            "repo": str(tmp_path / "mole_repo"), "python": "/p", "commit": head,
            "snapshot": str(snapshot),
            "checkpoint_id": f"Maple728/TimeMoE-50M@{revision}",
        },
    }

    observed = final_comparison.verify_external_versions(external, ["mole", "time_moe"])
    assert observed["mole"]["repo_head"] == head
    assert observed["time_moe"]["snapshot_revision"] == revision

    # 仓库实际处在另一个版本
    wrong = {**external, "mole": {**external["mole"], "commit": "0" * 40}}
    with pytest.raises(FinalComparisonError, match="HEAD"):
        final_comparison.verify_external_versions(wrong, ["mole"])

    # 声称一个 checkpoint_id，实际加载另一个 snapshot
    other = {**external, "time_moe": {
        **external["time_moe"],
        "checkpoint_id": "Maple728/TimeMoE-50M@" + "1" * 40,
    }}
    with pytest.raises(FinalComparisonError, match="snapshot"):
        final_comparison.verify_external_versions(other, ["time_moe"])

    # checkpoint_id 必须带不可变版本号，不能只写 repo id 或分支名
    bare = {**external, "time_moe": {
        **external["time_moe"], "checkpoint_id": "Maple728/TimeMoE-50M",
    }}
    with pytest.raises(FinalComparisonError, match="不可变"):
        final_comparison.verify_external_versions(bare, ["time_moe"])


def test_branch_name_is_not_an_immutable_revision(tmp_path):
    """``@main`` 加一个叫 main 的目录不算版本固定，即使路径根本不存在也会被放行。"""
    head = _git_repo(tmp_path / "repo")
    external = {"time_moe": {
        "repo": str(tmp_path / "repo"), "python": "/p", "commit": head,
        "snapshot": "/fake/snapshots/main",
        "checkpoint_id": "Maple728/TimeMoE-50M@main",
    }}

    with pytest.raises(FinalComparisonError, match="不可变"):
        final_comparison.verify_external_versions(external, ["time_moe"])


def test_missing_snapshot_directory_is_rejected(tmp_path):
    """revision 格式对但目录不存在时也必须拒绝：核对的是实际权重，不是字符串。"""
    head = _git_repo(tmp_path / "repo")
    revision = "b" * 40
    external = {"time_moe": {
        "repo": str(tmp_path / "repo"), "python": "/p", "commit": head,
        "snapshot": str(tmp_path / "missing" / "snapshots" / revision),
        "checkpoint_id": f"Maple728/TimeMoE-50M@{revision}",
    }}

    with pytest.raises(FinalComparisonError, match="不存在"):
        final_comparison.verify_external_versions(external, ["time_moe"])


def test_uncommitted_changes_to_tracked_sources_are_rejected(tmp_path):
    """HEAD 相等不代表跑的是那份代码：受跟踪源码被改过就必须拒绝。

    实验产物是未跟踪文件（训练/查询 CSV、mc_output 下的 npy），不应该因此失败。
    """
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    external = {"mole": {
        "repo": str(repo), "python": "/p", "checkpoints": "/c", "commit": head,
    }}

    # 未跟踪的实验产物不影响核对
    (repo / "mc_output").mkdir()
    (repo / "mc_output" / "real_prediction_pjm_T1_h24_s42.npy").write_text("x", encoding="utf-8")
    assert final_comparison.verify_external_versions(external, ["mole"])["mole"]["repo_head"] == head

    # 受跟踪源码被改过
    (repo / "a.txt").write_text("modified", encoding="utf-8")
    with pytest.raises(FinalComparisonError, match="未提交"):
        final_comparison.verify_external_versions(external, ["mole"])


def test_missing_local_paths_raise_a_handled_error_not_keyerror(tmp_path):
    """只有冻结定义、忘了传 --external-config 时，必须是可处理的错误而不是 KeyError。"""
    frozen = {"mole": {"commit": "0" * 40, "hyperparameters": {"seq_len": 96}}}
    merged = final_comparison.apply_frozen_external(None, frozen)

    with pytest.raises(FinalComparisonError, match="external-config"):
        final_comparison.verify_external_versions(merged, ["mole"])

    with pytest.raises(FinalComparisonError, match="snapshot"):
        final_comparison.verify_external_versions(
            {"time_moe": {"repo": str(tmp_path), "python": "/p", "commit": "0" * 40,
                          "checkpoint_id": "x@" + "c" * 40}},
            ["time_moe"],
        )
