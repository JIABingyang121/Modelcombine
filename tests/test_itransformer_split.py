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
BATCH_SIZE = 16
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
    (PJM_ROWS, 720, {"train": 4596, "val": 735, "test": 1332}),
    # VIC / NSW 3695
    (AEMO_ROWS, 24, {"train": 2586, "val": 370, "test": 739}),
    (AEMO_ROWS, 168, {"train": 2586, "val": 370, "test": 739}),
    (AEMO_ROWS, 720, {"train": 2221, "val": 735, "test": 739}),
])
def test_exact_split_sizes_for_the_formal_row_counts(rows, pred_len, expected):
    sizes = itransformer_split_sizes(rows, pred_len, batch_size=BATCH_SIZE)

    assert {k: sizes[k] for k in ("train", "val", "test")} == expected
    assert sizes["train"] + sizes["val"] + sizes["test"] == rows, "三段必须恰好覆盖全部行"


@pytest.mark.parametrize("rows", [PJM_ROWS, AEMO_ROWS])
def test_short_horizons_keep_the_official_boundaries(rows):
    """官方验证段够长时，切分必须与官方 70/10/20 逐个数字相同。"""
    for pred_len in (24, 168):
        sizes = itransformer_split_sizes(rows, pred_len, batch_size=BATCH_SIZE)
        assert sizes["val"] == sizes["official_val"], "验证段不该被动过"
        assert sizes["test"] == rows * 20 // 100
        assert sizes["train"] == rows * 70 // 100


def test_only_the_long_horizon_widens_validation():
    pjm_short = itransformer_split_sizes(PJM_ROWS, 168, batch_size=BATCH_SIZE)
    pjm_long = itransformer_split_sizes(PJM_ROWS, 720, batch_size=BATCH_SIZE)

    assert pjm_long["val"] > pjm_short["val"]
    assert pjm_long["train"] < pjm_short["train"]
    assert pjm_long["test"] == pjm_short["test"], "测试段不得因预测长度改变"


# ----------------------------------------------------- loader 理论样本数判据
@pytest.mark.parametrize("rows", [PJM_ROWS, AEMO_ROWS])
@pytest.mark.parametrize("pred_len", [24, 168, 720])
def test_every_loader_yields_at_least_one_sample(rows, pred_len):
    sizes = itransformer_split_sizes(rows, pred_len, batch_size=BATCH_SIZE)
    samples = itransformer_loader_samples(sizes, seq_len=SEQ_LEN, pred_len=pred_len)

    assert all(v >= 1 for v in samples.values()), samples
    # 判据必须与官方 __len__ 一致：val/test 文件各前置 seq_len 行上下文
    assert samples["val"] == (SEQ_LEN + sizes["val"]) - SEQ_LEN - pred_len + 1
    assert samples["test"] == (SEQ_LEN + sizes["test"]) - SEQ_LEN - pred_len + 1


def test_old_fixed_ratio_would_have_produced_a_negative_length():
    """复现现场：官方固定切分在 PJM × 720 上算出 -52。"""
    sizes = itransformer_split_sizes(PJM_ROWS, 720, batch_size=BATCH_SIZE)
    official_val_samples = sizes["official_val"] - 720 + 1

    assert official_val_samples == -52, "装置无效：没有复现出现场的负长度"
    assert itransformer_loader_samples(
        sizes, seq_len=SEQ_LEN, pred_len=720)["val"] == BATCH_SIZE


# --------------------------------------------------------------- 写出三份 CSV
def test_writes_three_files_with_context_rows_only_as_input(tmp_path):
    frame = _series(PJM_ROWS)

    plan = write_itransformer_splits(
        frame, tmp_path / "split", seq_len=SEQ_LEN, pred_len=720, batch_size=BATCH_SIZE
    )

    assert plan["policy"] == ITRANSFORMER_SPLIT_POLICY
    assert plan["file_rows"] == {
        "train": 4596, "val": SEQ_LEN + 735, "test": SEQ_LEN + 1332,
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
    write_itransformer_splits(frame, tmp_path / "s", seq_len=SEQ_LEN, pred_len=720, batch_size=BATCH_SIZE)

    last = pd.Timestamp(frame["timestamp"].max())
    for name in ("train", "val", "test"):
        stamps = pd.to_datetime(pd.read_csv(tmp_path / "s" / f"{name}.csv")["date"])
        assert stamps.max() <= last
        assert stamps.min() >= pd.Timestamp(frame["timestamp"].min())
        assert stamps.is_monotonic_increasing


def test_refuses_before_launching_official_training_when_too_short(tmp_path):
    """任何一个 loader 产不出样本就提前拒绝，不让官方在 __len__ 返回负数时抛栈。"""
    with pytest.raises(ExternalAdapterError, match="样本不足"):
        write_itransformer_splits(
            _series(900), tmp_path / "tiny", seq_len=SEQ_LEN, pred_len=720,
            batch_size=BATCH_SIZE,
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


# ------------------------- 官方 val 走 drop_last=True，必须凑满一个完整 batch
@pytest.mark.parametrize("rows", [PJM_ROWS, AEMO_ROWS])
def test_long_horizon_validation_yields_exactly_one_full_batch(rows):
    """官方 data_factory 的 val 走 else 分支：drop_last=True、batch_size=args.batch_size。

    样本数少于一个 batch 时整批被丢弃，验证集实际为空——所以下限是 pred_len+batch_size-1，
    不是 pred_len。
    """
    sizes = itransformer_split_sizes(rows, 720, batch_size=BATCH_SIZE)
    samples = itransformer_loader_samples(sizes, seq_len=SEQ_LEN, pred_len=720)

    assert sizes["val"] == 720 + BATCH_SIZE - 1 == 735
    assert samples["val"] == BATCH_SIZE, "验证样本必须恰好凑满一个 batch"
    # drop_last=True 下的实际批数
    assert samples["val"] // BATCH_SIZE == 1
    assert samples["train"] // BATCH_SIZE >= 1


def test_pred_len_only_lower_bound_would_have_emptied_validation():
    """反向确认下限必须含 batch_size：只保证 val_samples>=1 时 drop_last 会丢光整批。"""
    sizes_wrong = {"val": 720}                      # 旧下限 n_val = pred_len
    val_samples_wrong = sizes_wrong["val"] - 720 + 1

    assert val_samples_wrong == 1
    assert val_samples_wrong // BATCH_SIZE == 0, "drop_last=True 下这会得到 0 批"


# --------------------------------------------------- 命令必须用官方已有的 custom
def test_training_command_uses_official_custom_and_the_wrapper(tmp_path):
    from src.models.external_adapters import (
        ITRANSFORMER_TRAIN_SNIPPET,
        itransformer_command,
        itransformer_predict_command,
    )

    config = {
        "repo": tmp_path / "repo", "python": tmp_path / "py",
        "checkpoints": tmp_path / "ckpt", "gpu": 0,
        "hyperparameters": {
            "seq_len": 96, "label_len": 48, "d_model": 512, "n_heads": 8, "e_layers": 3,
            "d_layers": 1, "d_ff": 512, "factor": 1, "dropout": 0.1, "train_epochs": 10,
            "batch_size": 16, "patience": 3, "learning_rate": 0.0001, "des": "final",
            "split_policy": ITRANSFORMER_SPLIT_POLICY,
        },
    }
    train = itransformer_command(
        config, model_id="pjm_h720", data_path="train.csv", forecast_steps=720,
        root_path=tmp_path / "split",
    )
    query = itransformer_predict_command(
        config, model_id="pjm_h720", data_path="q.csv", forecast_steps=720,
        root_path=tmp_path / "split", output=tmp_path / "o.npy",
    )

    for command in (train, query):
        assert command[command.index("--data") + 1] == "custom"
        assert "custom_fixed" not in command, "锁定的官方 data_dict 没有这个键"
    # 训练走包装入口，不再直接执行 run.py
    assert train[1] == "-c" and train[2] == ITRANSFORMER_TRAIN_SNIPPET
    assert "run.py" not in train


# ------------------- 用模拟锁定官方仓库的桩，真跑一次包装入口（不需要 torch 全家桶）
_STUB_FACTORY = '''
class Dataset_Custom:
    def __init__(self, **kwargs):
        raise AssertionError("官方 Dataset_Custom 不该被使用——应已被固定切分替换")

data_dict = {"custom": Dataset_Custom}

def data_provider(args, flag):
    Data = data_dict[args.data]
    return Data(root_path=args.root_path, data_path=args.data_path, flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features, target=args.target, timeenc=1, freq=args.freq), None
'''

_STUB_RUN = '''
if __name__ == "__main__":
    import argparse, json
    from data_provider.data_factory import data_provider, data_dict
    parser = argparse.ArgumentParser()
    for flag in ("--data", "--root_path", "--data_path", "--features", "--target", "--freq"):
        parser.add_argument(flag)
    for flag in ("--seq_len", "--label_len", "--pred_len"):
        parser.add_argument(flag, type=int)
    args, _rest = parser.parse_known_args()
    assert set(data_dict) == {"custom"}, data_dict
    sizes = {}
    for flag in ("train", "val", "test"):
        dataset, _ = data_provider(args, flag)
        sizes[flag] = len(dataset)
    open("wrapper_result.json", "w").write(json.dumps(sizes))
'''


def test_wrapper_replaces_the_dataset_inside_a_locked_official_repo(tmp_path):
    """桩仓库的 data_dict 初始只有 custom（与锁定的 c2426e68 一致）。

    包装入口必须能在不改仓库文件的前提下把它换掉，run.py 照常执行，不再出现 KeyError，
    也不会落到官方 Dataset_Custom 的比例切分上。
    """
    import json
    import subprocess
    import sys

    from src.models.external_adapters import (
        ITRANSFORMER_TRAIN_SNIPPET,
        write_itransformer_splits,
    )

    repo = tmp_path / "official"
    (repo / "data_provider").mkdir(parents=True)
    (repo / "data_provider" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "data_provider" / "data_factory.py").write_text(_STUB_FACTORY, encoding="utf-8")
    (repo / "utils").mkdir()
    (repo / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "utils" / "timefeatures.py").write_text(
        "import numpy as np\n"
        "def time_features(dates, freq='h'):\n"
        "    assert hasattr(dates, 'hour'), type(dates)\n"
        "    return np.zeros((4, len(dates)))\n",
        encoding="utf-8",
    )
    (repo / "run.py").write_text(_STUB_RUN, encoding="utf-8")

    split_root = repo / "split"
    plan = write_itransformer_splits(
        _series(PJM_ROWS), split_root, seq_len=SEQ_LEN, pred_len=720,
        batch_size=BATCH_SIZE,
    )
    before = {p: p.read_bytes() for p in sorted(repo.rglob("*.py"))}

    completed = subprocess.run(
        [sys.executable, "-c", ITRANSFORMER_TRAIN_SNIPPET,
         "--data", "custom", "--root_path", f"{split_root}/", "--data_path", "train.csv",
         "--features", "S", "--target", "load", "--freq", "h",
         "--seq_len", str(SEQ_LEN), "--label_len", "48", "--pred_len", "720"],
        cwd=repo, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "KeyError" not in completed.stderr
    sizes = json.loads((repo / "wrapper_result.json").read_text())
    assert sizes == plan["loader_samples"], "替换后的 __len__ 必须与预算的样本数一致"
    assert sizes["val"] == BATCH_SIZE

    after = {p: p.read_bytes() for p in sorted(repo.rglob("*.py"))}
    assert after == before, "外部仓库源码文件在运行前后必须字节不变"
