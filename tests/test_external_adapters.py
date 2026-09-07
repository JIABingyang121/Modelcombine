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
    METHOD_SEEDS,
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
    command = itransformer_command(
        config, model_id="pjm_T1_h720", data_path="mc.csv", forecast_steps=720
    )
    assert command[2] == "run.py"
    for flag in ("--is_training", "--do_predict", "--inverse", "--pred_len"):
        assert flag in command
    assert command[command.index("--pred_len") + 1] == "720"
    assert command[command.index("--model") + 1] == "iTransformer"
    # 官方 run.py 没有种子参数，探针记录内部固定 2023
    assert "--seed" not in command
    assert METHOD_SEEDS["itransformer"] == 2023

    mole = mole_train_command(
        {**config, "repo": tmp_path / "mole"},
        model_id="pjm_T1_h24", data_path="mc.csv", forecast_steps=24,
    )
    assert mole[2] == "run_longExp.py"
    assert mole[mole.index("--model") + 1] == "MoLE_DLinear"
    assert mole[mole.index("--seed") + 1] == "42"
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
