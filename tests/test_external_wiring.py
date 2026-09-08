"""外部方法经统一入口跑 T1—T3 的真实接线。

已有的 test_external_adapters 只覆盖输入辅助函数、命令构造和输出解析；本文件覆盖
**接线**：训练文件与当前查询历史必须分开、训练按 dataset×forecast_steps×seed 只做一次
并在 T1—T3 复用、请求的种子必须真正传给官方方法。

subprocess 调用本身仍不在本机测（没有官方仓库也没有 GPU）；这里把官方调用替换成记录
调用参数的探针，验证的是**我们喂给官方什么**，不是官方内部行为。
"""
from __future__ import annotations

import json
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
            },
        }
        for method in ("itransformer", "mole", "time_moe")
    }

    calls = []

    def _record(command, *, cwd, method):
        calls.append({"method": method, "command": [str(c) for c in command]})

    reads = []

    def _fake_output(path, method, forecast_steps):
        reads.append(str(path))
        return np.full(forecast_steps, 1000.0)

    monkeypatch.setattr(final_comparison, "run_official", _record)
    monkeypatch.setattr(final_comparison, "read_official_output", _fake_output)
    monkeypatch.setattr(
        final_comparison, "_locate_checkpoint", lambda *a, **k: Path("ckpt.pth")
    )
    return {
        "raw_root": raw_root, "plan": plan, "external": external,
        "calls": calls, "reads": reads, "repo": repo,
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

    training_files = _written(wiring["repo"], "mc_train_*.csv")
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

    trains = [c for c in wiring["calls"] if "run_longExp.py" in c["command"]]
    predicts = [c for c in wiring["calls"] if "run_longExp.py" not in c["command"]]
    assert len(trains) == 1, f"应只训练一次，实际 {len(trains)} 次"
    assert len(predicts) == len(WINDOWS)


# ------------------------------------------------------------ P0-2 真实种子
def test_requested_seed_reaches_the_official_command(wiring):
    """请求 43 就必须真的把 43 传给官方，而不是永远用 42。"""
    _run(wiring, "mole", seeds=(43,), windows=("T1",))

    train = next(c for c in wiring["calls"] if "run_longExp.py" in c["command"])
    assert train["command"][train["command"].index("--seed") + 1] == "43"
    predict = next(c for c in wiring["calls"] if "run_longExp.py" not in c["command"])
    assert "43" in predict["command"], "预测调用也必须收到请求的种子"


def test_different_seeds_do_not_share_artifact_paths(wiring):
    """不同种子的训练文件与产物路径必须区分，否则后一个种子会覆盖前一个。"""
    _run(wiring, "mole", seeds=(42, 43), windows=("T1",))

    training_files = _written(wiring["repo"], "mc_train_*.csv")
    assert len(training_files) == 2
    assert len({p.name for p in training_files}) == 2
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

    trains = [c for c in wiring["calls"] if "run.py" in c["command"]]
    queries = [c for c in wiring["calls"] if "run.py" not in c["command"]]
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
    stale = wiring["repo"] / "dataset" / f"mc_train_{DATASET}_h{STEPS}_s42.csv"
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
