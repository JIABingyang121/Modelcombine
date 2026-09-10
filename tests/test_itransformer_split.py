"""iTransformer 的自适应训练/验证切分（正式运行 h=720 失败的修复）。

官方 `Dataset_Custom` 按固定 70/10/20 内部切分。PJM 的正式训练文件只有 6663 行，验证段
667 行 < pred_len=720，官方 loader 的 `__len__` 算出 `667 - 720 + 1 = -52`，直接抛
`ValueError: __len__() should return >= 0`。这不是偶发故障，是固定比例与长预测长度在短序列
上的必然不兼容。

修法是最小、确定性的：`n_val = max(official_val, pred_len)`，训练段相应缩短，其余一律不动。
h=24 与 h=168 的边界与官方完全一致，只有 h=720 会扩大验证段。
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.models.external_adapters import (
    ITRANSFORMER_SPLIT_POLICY,
    ExternalAdapterError,
    itransformer_loader_samples,
    itransformer_split_sizes,
    write_itransformer_splits,
)

SEQ_LEN = 96
PJM_ROWS = 6663
AEMO_ROWS = 3695


def _series(n: int, start: str = "2016-01-01 00:00:00") -> pd.DataFrame:
    stamps = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({"timestamp": stamps, "load": range(n)})


# ------------------------------------------------------- 精确切分（正式行数）
@pytest.mark.parametrize("rows, pred_len, expected", [
    # PJM 6663：h=24/168 保持官方 70/10/20 边界；h=720 扩大验证段
    (PJM_ROWS, 24, {"train": 4664, "val": 667, "test": 1332}),
    (PJM_ROWS, 168, {"train": 4664, "val": 667, "test": 1332}),
    (PJM_ROWS, 720, {"train": 4611, "val": 720, "test": 1332}),
    # VIC / NSW 3695
    (AEMO_ROWS, 24, {"train": 2586, "val": 370, "test": 739}),
    (AEMO_ROWS, 168, {"train": 2586, "val": 370, "test": 739}),
    (AEMO_ROWS, 720, {"train": 2236, "val": 720, "test": 739}),
])
def test_exact_split_sizes_for_the_formal_row_counts(rows, pred_len, expected):
    sizes = itransformer_split_sizes(rows, pred_len)

    assert {k: sizes[k] for k in ("train", "val", "test")} == expected
    assert sizes["train"] + sizes["val"] + sizes["test"] == rows, "三段必须恰好覆盖全部行"


@pytest.mark.parametrize("rows", [PJM_ROWS, AEMO_ROWS])
def test_short_horizons_keep_the_official_boundaries(rows):
    """官方验证段够长时，切分必须与官方 70/10/20 逐个数字相同。"""
    for pred_len in (24, 168):
        sizes = itransformer_split_sizes(rows, pred_len)
        assert sizes["val"] == sizes["official_val"], "验证段不该被动过"
        assert sizes["test"] == rows * 20 // 100
        assert sizes["train"] == rows * 70 // 100


def test_only_the_long_horizon_widens_validation():
    pjm_short = itransformer_split_sizes(PJM_ROWS, 168)
    pjm_long = itransformer_split_sizes(PJM_ROWS, 720)

    assert pjm_long["val"] > pjm_short["val"]
    assert pjm_long["train"] < pjm_short["train"]
    assert pjm_long["test"] == pjm_short["test"], "测试段不得因预测长度改变"


# ----------------------------------------------------- loader 理论样本数判据
@pytest.mark.parametrize("rows", [PJM_ROWS, AEMO_ROWS])
@pytest.mark.parametrize("pred_len", [24, 168, 720])
def test_every_loader_yields_at_least_one_sample(rows, pred_len):
    sizes = itransformer_split_sizes(rows, pred_len)
    samples = itransformer_loader_samples(sizes, seq_len=SEQ_LEN, pred_len=pred_len)

    assert all(v >= 1 for v in samples.values()), samples
    # 判据必须与官方 __len__ 一致：val/test 文件各前置 seq_len 行上下文
    assert samples["val"] == (SEQ_LEN + sizes["val"]) - SEQ_LEN - pred_len + 1
    assert samples["test"] == (SEQ_LEN + sizes["test"]) - SEQ_LEN - pred_len + 1


def test_old_fixed_ratio_would_have_produced_a_negative_length():
    """复现现场：官方固定切分在 PJM × 720 上算出 -52。"""
    sizes = itransformer_split_sizes(PJM_ROWS, 720)
    official_val_samples = sizes["official_val"] - 720 + 1

    assert official_val_samples == -52, "装置无效：没有复现出现场的负长度"
    assert itransformer_loader_samples(
        sizes, seq_len=SEQ_LEN, pred_len=720)["val"] == 1


# --------------------------------------------------------------- 写出三份 CSV
def test_writes_three_files_with_context_rows_only_as_input(tmp_path):
    frame = _series(PJM_ROWS)

    plan = write_itransformer_splits(
        frame, tmp_path / "split", seq_len=SEQ_LEN, pred_len=720
    )

    assert plan["policy"] == ITRANSFORMER_SPLIT_POLICY
    assert plan["file_rows"] == {
        "train": 4611, "val": SEQ_LEN + 720, "test": SEQ_LEN + 1332,
    }
    train = pd.read_csv(tmp_path / "split" / "train.csv")
    val = pd.read_csv(tmp_path / "split" / "val.csv")
    test = pd.read_csv(tmp_path / "split" / "test.csv")
    assert list(train.columns) == ["date", "load"], "官方 loader 要 date 列"

    # 前置上下文只作输入：val 的前 seq_len 行正是 train 的末尾 seq_len 行
    assert val["date"].iloc[:SEQ_LEN].tolist() == train["date"].iloc[-SEQ_LEN:].tolist()
    # 目标段本身互不重叠，且首尾相接
    assert val["date"].iloc[SEQ_LEN] > train["date"].iloc[-1]
    assert test["date"].iloc[SEQ_LEN] > val["date"].iloc[-1]
    assert test["date"].iloc[:SEQ_LEN].tolist() == val["date"].iloc[-SEQ_LEN:].tolist()


def test_all_rows_stay_within_the_input_frame(tmp_path):
    """切分只重排已有行，不得产生输入范围之外的时间戳。"""
    frame = _series(AEMO_ROWS)
    write_itransformer_splits(frame, tmp_path / "s", seq_len=SEQ_LEN, pred_len=720)

    last = pd.Timestamp(frame["timestamp"].max())
    for name in ("train", "val", "test"):
        stamps = pd.to_datetime(pd.read_csv(tmp_path / "s" / f"{name}.csv")["date"])
        assert stamps.max() <= last
        assert stamps.min() >= pd.Timestamp(frame["timestamp"].min())
        assert stamps.is_monotonic_increasing


def test_refuses_before_launching_official_training_when_too_short(tmp_path):
    """任何一个 loader 产不出样本就提前拒绝，不让官方在 __len__ 返回负数时抛栈。"""
    with pytest.raises(ExternalAdapterError, match="产不出样本|不足"):
        write_itransformer_splits(
            _series(900), tmp_path / "tiny", seq_len=SEQ_LEN, pred_len=720
        )
    assert not (tmp_path / "tiny" / "train.csv").exists(), "拒绝时不得留下半份切分"


def test_declared_split_policy_must_match_the_implementation():
    """切分策略必须由配置显式声明并与实现一致——它决定训练/验证边界，不能只是个注释。"""
    from src.models.external_adapters import hyperparameters

    base = {
        "seq_len": 96, "label_len": 48, "d_model": 512, "n_heads": 8, "e_layers": 3,
        "d_layers": 1, "d_ff": 512, "factor": 1, "dropout": 0.1, "train_epochs": 10,
        "batch_size": 32, "patience": 3, "learning_rate": 0.0001, "des": "final",
    }

    with pytest.raises(ExternalAdapterError, match="split_policy"):
        hyperparameters({"hyperparameters": dict(base)}, "itransformer")

    with pytest.raises(ExternalAdapterError, match="当前实现是"):
        hyperparameters(
            {"hyperparameters": {**base, "split_policy": "fixed_70_10_20"}},
            "itransformer",
        )

    resolved = hyperparameters(
        {"hyperparameters": {**base, "split_policy": ITRANSFORMER_SPLIT_POLICY}},
        "itransformer",
    )
    assert resolved["split_policy"] == ITRANSFORMER_SPLIT_POLICY
