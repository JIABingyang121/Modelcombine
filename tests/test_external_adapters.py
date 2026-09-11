"""外部官方实现的适配器（src/models/external_adapters.py）。

官方仓库在 Modelcombine 之外、各自带独立 venv，本机既没有仓库也没有 GPU，因此**不测
subprocess 调用本身**。能在本机确定的是三件事，全部按 2026-09-08 服务器探针的实测契约验证：

1. 输入准备：写出官方要的 ``date,load``，物理截断到 training_cutoff，并断言最大时间戳；
2. 命令构造：与探针 command.txt 的官方入口一致；
3. 输出解析：用探针**真实带回的** ``real_prediction_<H>.npy`` 解析，形状不符即报错。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.external_adapters import (
    METHOD_LIMITATIONS,
    OFFICIAL_FIXED_SEED,
    ExternalAdapterError,
    itransformer_command,
    mole_train_command,
    read_official_output,
    write_official_input,
)

#: 服务器探针带回的真实官方输出
PROBE_ROOT = Path("/home/jia/桌面/modelcombine_stage1_20260908/probes")
FORECAST_STEPS = (24, 168, 720)


def _history(rows: int = 900, end: str = "2016-11-12 02:00:00") -> pd.DataFrame:
    stamps = pd.date_range(end=pd.Timestamp(end), periods=rows, freq="h")
    return pd.DataFrame({"timestamp": stamps, "load": np.linspace(20000, 30000, rows)})


# ------------------------------------------------------------------ 输入准备
def test_input_is_written_in_official_columns_and_truncated_at_cutoff(tmp_path):
    cutoff = pd.Timestamp("2016-11-12 02:00:00")
    history = _history()
    # 追加 cutoff 之后的数据：必须被物理截断掉，不能进入官方输入
    future = pd.DataFrame({
        "timestamp": pd.date_range("2016-11-12 03:00:00", periods=50, freq="h"),
        "load": np.full(50, 99999.0),
    })
    destination = tmp_path / "dataset" / "mc_probe.csv"

    write_official_input(pd.concat([history, future]), destination, cutoff)

    frame = pd.read_csv(destination)
    assert list(frame.columns) == ["date", "load"]
    assert pd.to_datetime(frame["date"]).max() == cutoff
    assert 99999.0 not in set(frame["load"]), "cutoff 之后的数据必须被截断"


def test_input_preparation_refuses_when_cutoff_is_not_reached(tmp_path):
    """历史不到 cutoff 时必须报错——否则外部方法会拿到一个错误的训练截止。"""
    history = _history(end="2016-11-01 00:00:00")
    with pytest.raises(ExternalAdapterError, match="与要求的 cutoff"):
        write_official_input(
            history, tmp_path / "d.csv", pd.Timestamp("2016-11-12 02:00:00")
        )

    with pytest.raises(ExternalAdapterError, match="没有可用历史"):
        write_official_input(
            history, tmp_path / "d.csv", pd.Timestamp("2000-01-01 00:00:00")
        )


# ------------------------------------------------------------------ 输出解析
@pytest.mark.parametrize("method", ["itransformer", "mole", "time_moe"])
@pytest.mark.parametrize("steps", FORECAST_STEPS)
def test_real_official_outputs_parse_to_a_flat_trajectory(method, steps):
    path = PROBE_ROOT / method / "native_output" / f"real_prediction_{steps}.npy"
    if not path.exists():
        pytest.skip(f"本机没有探针产物 {path}")

    values = read_official_output(path, method, steps)

    assert values.shape == (steps,)
    assert np.isfinite(values).all()
    # 探针输出已反归一化到负荷量级（PJM 约 2 万—3.5 万）
    assert 1e4 < float(values.min()) and float(values.max()) < 5e4


@pytest.mark.parametrize("method", ["itransformer", "mole", "time_moe"])
def test_unexpected_output_shape_is_rejected_not_squeezed(tmp_path, method):
    """形状与探针记录不符时必须报错，不能靠 squeeze 蒙混过去。"""
    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.zeros((1, 5, 1, 1), dtype=np.float32))
    with pytest.raises(ExternalAdapterError, match="形状"):
        read_official_output(wrong, method, 5)

    # 另一方法的形状同样不接受：itransformer 期望 (1,H,1)，time_moe 期望 (1,H)
    other = tmp_path / "other.npy"
    np.save(other, np.zeros((1, 5) if method != "time_moe" else (1, 5, 1), dtype=np.float32))
    with pytest.raises(ExternalAdapterError, match="形状"):
        read_official_output(other, method, 5)


def test_missing_or_non_finite_output_is_rejected(tmp_path):
    with pytest.raises(ExternalAdapterError, match="官方未产出"):
        read_official_output(tmp_path / "absent.npy", "mole", 24)

    bad = tmp_path / "bad.npy"
    np.save(bad, np.full((1, 4, 1), np.nan, dtype=np.float32))
    with pytest.raises(ExternalAdapterError, match="非有限值"):
        read_official_output(bad, "mole", 4)


# ------------------------------------------------------------------ 命令构造
def test_commands_match_the_probed_official_entries(tmp_path):
    config = {
        "repo": tmp_path / "iTransformer", "python": tmp_path / "py",
        "checkpoints": tmp_path / "ckpt", "gpu": 0,
    }
    config["hyperparameters"] = {
        "seq_len": 96, "label_len": 48, "d_model": 512, "n_heads": 8, "e_layers": 3,
        "d_layers": 1, "d_ff": 512, "factor": 1, "dropout": 0.1, "train_epochs": 10,
        "batch_size": 32, "patience": 3, "learning_rate": 0.0001, "t_dim": 4,
        "des": "final", "split_policy": "adaptive_val_at_least_one_full_batch",
    }
    command = itransformer_command(
        config, model_id="pjm_T1_h720", data_path="train.csv", forecast_steps=720,
        root_path=tmp_path / "split",
    )
    # 训练走包装入口（运行时替换 data_dict["custom"]），不再直接执行 run.py
    from src.models.external_adapters import ITRANSFORMER_TRAIN_SNIPPET
    assert command[1] == "-c" and command[2] == ITRANSFORMER_TRAIN_SNIPPET
    for flag in ("--is_training", "--inverse", "--pred_len"):
        assert flag in command
    # 训练阶段不带 --do_predict：它产出的 real_prediction.npy 与任何查询窗口都无关
    assert "--do_predict" not in command
    assert command[command.index("--pred_len") + 1] == "720"
    assert command[command.index("--model") + 1] == "iTransformer"
    # 官方 run.py 没有种子参数，探针记录内部固定 2023
    assert "--seed" not in command
    assert OFFICIAL_FIXED_SEED["itransformer"] == 2023

    mole = mole_train_command(
        {**config, "repo": tmp_path / "mole"},
        model_id="pjm_T1_h24", data_path="train.csv", forecast_steps=24, seed=43,
        root_path=tmp_path / "mole_split",
    )
    from src.models.external_adapters import MOLE_TRAIN_SNIPPET
    assert mole[1:4] == ["-u", "-c", MOLE_TRAIN_SNIPPET]
    assert mole[mole.index("--model") + 1] == "MoLE_DLinear"
    assert mole[mole.index("--root_path") + 1] == f"{tmp_path / 'mole_split'}/"
    assert mole[mole.index("--seed") + 1] == "43", "请求的种子必须真的传下去"
    # 官方 --do_predict 有真实缺陷，训练命令里不得带它
    assert "--do_predict" not in mole


def test_time_moe_pretraining_limitation_is_recorded():
    """Time-MoE 的预训练截止不可证，必须显式暴露，不能宣称满足严格训练截止约束。"""
    assert "time_moe" in METHOD_LIMITATIONS
    limitation = METHOD_LIMITATIONS["time_moe"]
    assert "预训练" in limitation and "不满足严格的训练截止" in limitation
    # 只有 Time-MoE 有这条限制；另外两个是本地训练的，训练截止可证
    assert "itransformer" not in METHOD_LIMITATIONS
    assert "mole" not in METHOD_LIMITATIONS


def test_formal_hyperparameters_are_required_and_probe_values_are_separate(tmp_path):
    """探针的 1 epoch 缩小配置不得作为正式默认值：缺 hyperparameters 直接失败。"""
    from src.models.external_adapters import (
        PROBE_HYPERPARAMETERS,
        hyperparameters,
    )

    bare = {"repo": tmp_path, "python": "py", "checkpoints": tmp_path}
    with pytest.raises(ExternalAdapterError, match="hyperparameters"):
        itransformer_command(bare, model_id="m", data_path="d.csv", forecast_steps=24,
                             root_path=tmp_path / "split")
    with pytest.raises(ExternalAdapterError, match="缺少"):
        hyperparameters({"hyperparameters": {"seq_len": 96}}, "itransformer")

    # 探针配置仍然可查，但必须显式传入才生效
    assert PROBE_HYPERPARAMETERS["train_epochs"] == 1
    assert PROBE_HYPERPARAMETERS["des"] == "probe"


def test_itransformer_query_command_carries_do_predict_and_the_same_setting(tmp_path):
    """查询命令：``python -c <片段> <产物路径>`` + 与训练完全相同的 setting 参数。"""
    from src.models.external_adapters import (
        ITRANSFORMER_PREDICT_SNIPPET,
        itransformer_predict_command,
    )

    config = {
        "repo": tmp_path / "iTransformer", "python": tmp_path / "py",
        "checkpoints": tmp_path / "ckpt", "gpu": 0,
        "hyperparameters": {
            "seq_len": 96, "label_len": 48, "d_model": 512, "n_heads": 8, "e_layers": 3,
            "d_layers": 1, "d_ff": 512, "factor": 1, "dropout": 0.1, "train_epochs": 10,
            "batch_size": 32, "patience": 3, "learning_rate": 0.0001, "des": "final",
            "split_policy": "adaptive_val_at_least_one_full_batch",
        },
    }
    train = itransformer_command(
        config, model_id="pjm_h24", data_path="train.csv", forecast_steps=24,
        root_path=tmp_path / "split",
    )
    query = itransformer_predict_command(
        config, model_id="pjm_h24", data_path="mc_pred_T1.csv", forecast_steps=24,
        root_path=tmp_path / "split", output=tmp_path / "out.npy",
    )

    assert query[1] == "-c" and query[2] == ITRANSFORMER_PREDICT_SNIPPET
    assert query[3] == str(tmp_path / "out.npy")
    assert "--do_predict" in query
    # 官方 setting 不含 data_path：训练与查询只有 --data_path 不同，才会落在同一个
    # checkpoint 上；--is_training 必须仍是 1（0 分支只调 exp.test，不产 predict 结果）
    assert query[query.index("--is_training") + 1] == "1"
    def _without_data_path(command, start):
        rest = command[start:]
        index = rest.index("--data_path")
        return rest[:index] + rest[index + 2:]
    assert _without_data_path(query, 4) == (
        _without_data_path(train, 3) + ["--do_predict"]
    )


# 桩官方仓库：复刻 run.py 的控制流（argparse -> setting -> train/test/predict），
# 用来验证**我们的查询片段**本身——它必须跳过训练、只调 predict(load=True)，并把官方
# 产物另存到指定路径。桩里不含任何官方源码，也不冒充官方结果。
_STUB_EXP = '''
import os
import numpy as np


class Parent:
    def __init__(self, args):
        self.args = args


class Exp_Long_Term_Forecast(Parent):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.model = "model"

    def train(self, setting):
        open("train_ran", "a").close()
        return self.model

    def test(self, setting, test=0):
        open("test_ran", "a").close()

    def predict(self, setting, load=False):
        open("predict_ran", "a").write(f"{setting}|{load}\\n")
        folder = "./results/" + setting + "/"
        os.makedirs(folder, exist_ok=True)
        np.save(folder + "real_prediction.npy", np.arange(3.0).reshape(1, 3, 1))
'''

_STUB_RUN = '''
if __name__ == "__main__":
    import argparse
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_training", type=int, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--do_predict", action="store_true")
    args, _rest = parser.parse_known_args()
    setting = args.model_id + "_iTransformer_custom_ftS_0"
    if args.is_training:
        exp = Exp_Long_Term_Forecast(args)
        exp.train(setting)
        exp.test(setting)
        if args.do_predict:
            exp.predict(setting, True)
    else:
        exp = Exp_Long_Term_Forecast(args)
        exp.test(setting, test=1)
'''


def test_predict_snippet_skips_training_and_saves_to_the_given_path(tmp_path):
    """真跑查询片段：不得触发 train/test，必须 predict(load=True) 并另存到指定路径。"""
    import subprocess
    import sys

    import numpy as np

    from src.models.external_adapters import ITRANSFORMER_PREDICT_SNIPPET

    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "experiments" / "exp_long_term_forecasting.py").write_text(
        _STUB_EXP, encoding="utf-8"
    )
    (repo / "run.py").write_text(_STUB_RUN, encoding="utf-8")
    out = tmp_path / "real_prediction_T1.npy"

    completed = subprocess.run(
        [sys.executable, "-c", ITRANSFORMER_PREDICT_SNIPPET, str(out),
         "--is_training", "1", "--model_id", "pjm_h24", "--do_predict"],
        cwd=repo, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (repo / "train_ran").exists(), "查询不得重新训练"
    assert not (repo / "test_ran").exists(), "查询不得跑官方 test 流程"
    assert (repo / "predict_ran").read_text().strip() == "pjm_h24_iTransformer_custom_ftS_0|True"
    assert np.load(out).shape == (1, 3, 1)


@pytest.mark.parametrize("method", ["itransformer", "mole"])
def test_run_official_sets_cuda_visibility_before_external_training_starts(
    tmp_path, monkeypatch, method,
):
    """官方 ECL 脚本在启动 Python 前 export GPU，避免 torch 导入后再缩减可见设备。"""
    import subprocess

    from src.models.external_adapters import run_official

    observed = {}

    def fake_run(command, *, cwd, capture_output, text, env):
        observed.update({"command": command, "cwd": cwd, "env": env})
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("src.models.external_adapters.subprocess.run", fake_run)
    run_official(
        ["python", "run.py", "--gpu", "2"], cwd=tmp_path,
        method=method,
    )

    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "2"
