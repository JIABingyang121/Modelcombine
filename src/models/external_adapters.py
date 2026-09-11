"""调用外部官方实现的适配器：输入准备、命令构造、输出解析。

三个官方项目放在 Modelcombine 仓库之外，各自带独立 venv（iTransformer/MoLE 用
torch 2.0.0，Time-MoE 用 torch 2.7.1），因此**只能以 subprocess 调用**，不能在本进程
import。本模块不含任何第三方源码，只负责三件事：

1. 把冻结窗口的历史写成官方加载器要的 CSV，并在物理上截断到 ``training_cutoff``；
2. 按服务器探针实测通过的命令构造调用；
3. 解析官方产出的 ``real_prediction_<H>.npy``。

命令与输出格式来自 2026-09-08 的服务器探针（`probes/*/probe_manifest.json`），不是推测。
"""
from __future__ import annotations

import os
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

#: 官方加载器要求的列名（探针 manifest 的 ``official_input_columns``）。
OFFICIAL_COLUMNS = ("date", "load")

#: 各方法官方输出的 ndarray 形状，``H`` 为预测长度。形状变了要显式报错，不静默 squeeze。
OUTPUT_SHAPES: Dict[str, str] = {
    "itransformer": "(1, H, 1)",
    "mole": "(1, H, 1)",
    "time_moe": "(1, H)",
}

#: 探针实测的固定种子。iTransformer 官方 run.py 无 CLI 种子参数，硬编码 2023；
#: 因此结果表里 iTransformer 的 seed 必须记官方真实值 2023，而不是请求值。
OFFICIAL_FIXED_SEED: Dict[str, int] = {"itransformer": 2023}

#: 探针用的超参数：1 epoch、缩小模型、des=probe。**只用于接口验证**，
#: 正式对比必须由 --external-config 显式给出，否则比较的是"缩小版外部模型"。
PROBE_HYPERPARAMETERS: Dict[str, Any] = {
    "seq_len": 96, "label_len": 48, "d_model": 32, "n_heads": 4,
    "e_layers": 1, "d_layers": 1, "d_ff": 64, "factor": 1, "dropout": 0.1,
    "train_epochs": 1, "batch_size": 64, "patience": 1, "learning_rate": 0.0001,
    "t_dim": 4, "des": "probe",
}

#: 正式命令必须齐备的超参数键。缺任何一个都直接失败，不用探针值兜底。
#: 外部训练方法的切分策略标识，写进 external_formal.json 的超参数，使冻结定义记录它。
ITRANSFORMER_SPLIT_POLICY = "adaptive_val_at_least_one_full_batch"
MOLE_SPLIT_POLICY = ITRANSFORMER_SPLIT_POLICY

REQUIRED_HYPERPARAMETERS: Dict[str, tuple] = {
    "itransformer": (
        "seq_len", "label_len", "d_model", "n_heads", "e_layers", "d_layers",
        "d_ff", "factor", "dropout", "train_epochs", "batch_size", "patience",
        "learning_rate", "des",
        # 切分策略必须由配置显式声明并随冻结定义落盘：它决定了训练/验证段的边界
        "split_policy",
    ),
    "mole": (
        "seq_len", "t_dim", "train_epochs", "batch_size", "patience",
        "learning_rate", "des", "split_policy",
    ),
}


#: 走官方外部实现的方法。它们的正式口径必须先进冻结定义，运行时只能用冻结值。
EXTERNAL_METHODS = ("itransformer", "mole", "time_moe")

#: 每个外部方法必须写进冻结定义的口径字段。``commit`` 是官方代码版本（由配置声明，
#: 本项目不代为核验）；``checkpoint_id`` 是 Time-MoE 的预训练权重标识。
#: repo/python/checkpoints/snapshot 这些是机器本地路径，不属于口径，不进定义。
FROZEN_EXTERNAL_KEYS: Dict[str, tuple] = {
    "itransformer": ("commit", "hyperparameters"),
    "mole": ("commit", "hyperparameters"),
    "time_moe": ("commit", "checkpoint_id", "context_length"),
}


#: 每个外部方法必须由 --external-config 提供的机器本地字段。冻结定义只存口径，不存路径，
#: 所以只有冻结定义而没有配置时，这些字段一个都拿不到。
REQUIRED_LOCAL_FIELDS: Dict[str, tuple] = {
    "itransformer": ("repo", "python", "checkpoints"),
    "mole": ("repo", "python", "checkpoints"),
    "time_moe": ("repo", "python", "snapshot"),
}


def hyperparameters(config: Mapping[str, Any], method: str) -> Dict[str, Any]:
    """取正式超参数；缺失即失败，绝不回落到 PROBE_HYPERPARAMETERS。"""
    values = config.get("hyperparameters")
    if not values:
        raise ExternalAdapterError(
            f"{method}: --external-config 必须提供 hyperparameters（正式对比配置）；"
            "探针的 1 epoch 缩小配置只用于接口验证，不能作为正式默认值"
        )
    missing = [k for k in REQUIRED_HYPERPARAMETERS[method] if k not in values]
    if missing:
        raise ExternalAdapterError(f"{method}: hyperparameters 缺少 {missing}")
    expected_policy = {
        "itransformer": ITRANSFORMER_SPLIT_POLICY,
        "mole": MOLE_SPLIT_POLICY,
    }.get(method)
    if expected_policy is not None and values["split_policy"] != expected_policy:
        raise ExternalAdapterError(
            f"{method}: 配置声明的 split_policy 是 {values['split_policy']!r}，"
            f"当前实现是 {expected_policy!r}"
        )
    return dict(values)

#: 必须随结果一起报告的方法级限制。Time-MoE 只能证明**任务输入上下文**的截止时间，
#: 公开预训练 checkpoint 无法证明其预训练数据早于该 cutoff。
METHOD_LIMITATIONS: Dict[str, str] = {
    "time_moe": (
        "公开预训练 checkpoint 不暴露也不保证其预训练数据截止于本实验的 training_cutoff；"
        "可证明的只有任务输入上下文的截止时间。因此 Time-MoE 不满足严格的训练截止可比性，"
        "与它的比较结果不得表述为同等训练截止条件下的比较。"
    ),
}


class ExternalAdapterError(RuntimeError):
    """外部方法未按契约产出结果。"""


def write_official_input(
    history: pd.DataFrame, destination: Path, training_cutoff: pd.Timestamp
) -> Path:
    """写出官方加载器要的 ``date,load`` CSV，并在物理上截断到 cutoff。

    探针的 ``training_cutoff_transport``：输入 CSV 物理截断，且调用前断言其最大时间戳
    等于 cutoff——这是把训练截止约束传递给外部实现的唯一手段。
    """
    frame = history[["timestamp", "load"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame[frame["timestamp"] <= pd.Timestamp(training_cutoff)]
    if frame.empty:
        raise ExternalAdapterError(f"截断到 {training_cutoff} 之后没有可用历史")
    observed = frame["timestamp"].max()
    if observed != pd.Timestamp(training_cutoff):
        raise ExternalAdapterError(
            f"输入 CSV 的最大时间戳是 {observed}，与要求的 cutoff {training_cutoff} 不符"
        )
    frame = frame.rename(columns={"timestamp": "date"})[list(OFFICIAL_COLUMNS)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def read_official_output(path: Path, method: str, forecast_steps: int) -> np.ndarray:
    """读取官方 ``real_prediction_<H>.npy`` 并校验形状，返回 ``(H,)``。"""
    if not path.exists():
        raise ExternalAdapterError(f"{method}: 官方未产出 {path}")
    array = np.load(path)
    expected = (1, forecast_steps, 1) if method in ("itransformer", "mole") else (1, forecast_steps)
    if array.shape != expected:
        raise ExternalAdapterError(
            f"{method}: 官方输出形状 {array.shape}，与探针记录的 "
            f"{OUTPUT_SHAPES[method].replace('H', str(forecast_steps))} 不符"
        )
    values = np.asarray(array, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ExternalAdapterError(f"{method}: 官方输出含非有限值")
    return values


def itransformer_split_sizes(
    n_rows: int, pred_len: int, *, batch_size: int
) -> Dict[str, int]:
    """自适应切分：官方 70/10/20 边界不变，只在验证段产不出一个完整 batch 时扩大它。

    官方 ``Dataset_Custom`` 按固定 70/10/20 内部切分，验证段长度 = N - floor(0.7N) - n_test。
    PJM 的正式训练文件只有 6663 行，验证段 667 行 < pred_len=720，官方 loader 的
    ``__len__`` 算出 ``667 - 720 + 1 = -52``，直接抛 ``ValueError: __len__() should
    return >= 0``。这不是偶发故障，是固定比例与长预测长度在短序列上的必然不兼容。

    仅仅让 val_samples >= 1 还不够：官方 ``data_factory`` 的 val 走 else 分支，
    ``drop_last=True`` 且 ``batch_size=args.batch_size``。样本数少于一个 batch 时整批被丢弃，
    验证集实际为空。因此下限取 ``pred_len + batch_size - 1``，使
    ``val_samples = n_val - pred_len + 1 >= batch_size``。

    h=24 与 h=168 的官方验证段本就够长，边界与原来逐个数字相同；只有 h=720 会扩大验证段。
    """
    n_rows = int(n_rows)
    pred_len = int(pred_len)
    batch_size = int(batch_size)
    n_test = n_rows * 20 // 100
    official_val = n_rows - (n_rows * 70 // 100) - n_test
    minimum_val = pred_len + batch_size - 1
    n_val = max(official_val, minimum_val)
    n_train = n_rows - n_val - n_test
    return {
        "rows": n_rows, "train": n_train, "val": n_val, "test": n_test,
        "official_val": official_val, "minimum_val": minimum_val,
        "pred_len": pred_len, "batch_size": batch_size,
    }


def itransformer_loader_samples(
    sizes: Mapping[str, int], *, seq_len: int, pred_len: int
) -> Dict[str, int]:
    """三个 loader 的理论样本数，判据与官方 ``__len__`` 一致。

    ``Dataset_Custom_Fixed.__len__ = len(rows) - seq_len - pred_len + 1``。验证段与测试段
    各自前置 ``seq_len`` 行历史上下文，那些重叠行只作输入、不属于目标段，所以它们的文件
    行数是 ``seq_len + n``。
    """
    seq_len, pred_len = int(seq_len), int(pred_len)
    return {
        "train": int(sizes["train"]) - seq_len - pred_len + 1,
        "val": int(sizes["val"]) - pred_len + 1,
        "test": int(sizes["test"]) - pred_len + 1,
    }


def write_itransformer_splits(
    frame: pd.DataFrame, destination: Path, *, seq_len: int, pred_len: int,
    batch_size: int,
) -> Dict[str, Any]:
    """按自适应切分写出官方 ``custom_fixed`` 需要的 train/val/test 三份 CSV。

    输入 ``frame`` 必须已经截断到正式 ``training_cutoff`` 之前——本函数不做截断，只切分。
    启动官方训练之前先算三个 loader 的理论样本数，任何一个 < 1 就直接失败，不让官方在
    ``__len__`` 返回负数时抛栈。
    """
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    sizes = itransformer_split_sizes(len(ordered), pred_len, batch_size=batch_size)
    samples = itransformer_loader_samples(sizes, seq_len=seq_len, pred_len=pred_len)
    # train 与 val 都走 drop_last=True 的分支，不足一个 batch 会被整批丢弃；test 的
    # batch_size 为 1、drop_last=False，只要求至少一个样本
    required = {"train": batch_size, "val": batch_size, "test": 1}
    short = {
        name: (samples[name], need)
        for name, need in required.items() if samples[name] < need
    }
    if short:
        raise ExternalAdapterError(
            f"itransformer: 切分后这些 loader 样本不足（实际, 需要）{short}——"
            f"行数={sizes['rows']} seq_len={seq_len} pred_len={pred_len} "
            f"batch_size={batch_size} "
            f"train/val/test={sizes['train']}/{sizes['val']}/{sizes['test']}"
        )

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    train_end = sizes["train"]
    val_end = train_end + sizes["val"]
    written: Dict[str, int] = {}
    for name, start, end in (
        ("train", 0, train_end),
        # 验证段与测试段各前置 seq_len 行输入上下文；这些行只作输入，不是目标
        ("val", train_end - seq_len, val_end),
        ("test", val_end - seq_len, val_end + sizes["test"]),
    ):
        part = ordered.iloc[start:end][["timestamp", "load"]].rename(
            columns={"timestamp": "date"}
        )
        part.to_csv(destination / f"{name}.csv", index=False)
        written[name] = int(len(part))
    return {
        "policy": ITRANSFORMER_SPLIT_POLICY,
        "sizes": sizes, "loader_samples": samples, "file_rows": written,
        "seq_len": int(seq_len), "batch_size": int(batch_size),
    }


def itransformer_flags(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int,
    do_predict: bool, root_path: Path,
) -> List[str]:
    """官方 ``run.py`` 的参数（不含解释器与脚本名）。

    官方的 ``setting``（也就是 checkpoint 目录名）由 model_id/model/data/features/三段
    长度/模型尺寸/des 拼成，**不含 data_path**。因此训练与查询用同一份参数、只换
    ``--data_path``，就落在同一个 checkpoint 上。
    """
    hp = hyperparameters(config, "itransformer")
    return [
        "--is_training", "1",
        "--model_id", model_id, "--model", "iTransformer",
        # 锁定的官方 c2426e68 的 data_dict 只有 custom。固定切分由训练包装入口在运行时
        # 把 data_dict["custom"] 换掉来提供，官方源码一行不改；预测走 flag='pred'，
        # 官方仍然选 Dataset_Pred，不受替换影响。
        "--data", "custom", "--root_path", f"{root_path}/", "--data_path", data_path,
        "--features", "S", "--target", "load", "--freq", "h",
        "--checkpoints", str(config["checkpoints"]),
        "--seq_len", str(hp["seq_len"]), "--label_len", str(hp["label_len"]),
        "--pred_len", str(forecast_steps),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--d_model", str(hp["d_model"]), "--n_heads", str(hp["n_heads"]),
        "--e_layers", str(hp["e_layers"]), "--d_layers", str(hp["d_layers"]),
        "--d_ff", str(hp["d_ff"]), "--factor", str(hp["factor"]),
        "--dropout", str(hp["dropout"]), "--embed", "timeF",
        "--train_epochs", str(hp["train_epochs"]), "--batch_size", str(hp["batch_size"]),
        "--patience", str(hp["patience"]), "--learning_rate", str(hp["learning_rate"]),
        "--num_workers", "0", "--itr", "1", "--des", str(hp["des"]),
        "--inverse", "--gpu", str(config.get("gpu", 0)),
    ] + (["--do_predict"] if do_predict else [])


#: 训练包装入口：在外部进程内把官方 ``data_dict["custom"]`` 换成固定切分 Dataset，
#: 再用 runpy 执行官方 ``run.py``。**官方仓库源码一行不改。**
#:
#: 为什么必须这样做：锁定的 c2426e68 的 ``data_dict`` 只有 ``custom``，没有
#: ``custom_fixed``；而 ``Dataset_Custom`` 按固定 70/10/20 内部切分，短序列 + 长
#: ``pred_len`` 必然算出负长度。替换 ``data_dict`` 的条目即可让官方模型、训练循环、
#: DataLoader 与预测逻辑全部保持原样，只换数据来源。
#:
#: ``data_provider()`` 内部是 ``Data = data_dict[args.data]``，在调用时才查模块全局字典，
#: 所以运行前改 ``factory.data_dict`` 生效；``exp_*`` 里 ``from ... import data_provider``
#: 绑定的是函数本身，不受影响。
ITRANSFORMER_TRAIN_SNIPPET = """
import os, runpy
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import data_provider.data_factory as factory


class FixedSplitDataset(Dataset):
    \"\"\"train/val/test 各读一份固定 CSV；scaler 统一在 train.csv 上拟合。

    __len__ 与官方 Dataset_Custom 完全一致；val/test 的 CSV 已包含 seq_len 行输入上下文。
    \"\"\"

    FILES = {'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv'}

    def __init__(self, root_path, flag='train', size=None, features='S',
                 data_path='train.csv', target='OT', scale=True, timeenc=0, freq='h'):
        assert flag in self.FILES, flag
        self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.__read_data__()

    def _read(self, name):
        frame = pd.read_csv(os.path.join(self.root_path, name))
        if 'date' not in frame.columns:
            for alt in ('timestamp', 'ds', 'datetime', 'time', 'ts'):
                if alt in frame.columns:
                    frame = frame.rename(columns={alt: 'date'})
                    break
        if 'date' not in frame.columns:
            raise ValueError('fixed split needs a date column: ' + name)
        return frame

    def __read_data__(self):
        self.scaler = StandardScaler()
        train = self._read('train.csv')
        split = self._read(self.FILES[self.flag])
        if self.features in ('M', 'MS'):
            train_values = train.drop(columns=['date'])
            split_values = split.drop(columns=['date'])
        else:
            train_values = train[[self.target]]
            split_values = split[[self.target]]
        if self.scale:
            self.scaler.fit(train_values.values)
            data = self.scaler.transform(split_values.values)
        else:
            data = split_values.values
        stamp = pd.to_datetime(split['date'])
        if self.timeenc == 0:
            data_stamp = pd.DataFrame({
                'month': stamp.dt.month, 'day': stamp.dt.day,
                'weekday': stamp.dt.weekday, 'hour': stamp.dt.hour,
            }).values
        else:
            data_stamp = time_features(pd.DatetimeIndex(stamp), freq=self.freq).transpose(1, 0)
        self.data_x = np.asarray(data, dtype=float)
        self.data_y = self.data_x
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end], self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


factory.data_dict['custom'] = FixedSplitDataset
runpy.run_path('run.py', run_name='__main__')
"""


def itransformer_command(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int,
    root_path: Path,
) -> List[str]:
    """官方训练入口（探针实测通过的 ``--is_training 1`` 分支）。

    这里**不带** ``--do_predict``：带上它，官方会用训练文件再跑一次 predict 并写下
    ``results/<setting>/real_prediction.npy``；那份文件与任何查询窗口都无关，正是旧接线
    误当成查询结果读回的东西。训练阶段干脆不产生它。
    """
    return [str(config["python"]), "-c", ITRANSFORMER_TRAIN_SNIPPET] + itransformer_flags(
        config, model_id=model_id, data_path=data_path, root_path=root_path,
        forecast_steps=forecast_steps, do_predict=False,
    )


#: iTransformer 查询：以官方 ``Exp.predict`` 加载已训练 checkpoint 预测当前窗口。
#:
#: **不能用** ``--is_training 0``：官方 ``run.py`` 的该分支只调 ``exp.test()``，根本不读
#: ``--do_predict``；而只有 ``exp.predict()`` 写 ``real_prediction.npy``（``test()`` 写的是
#: ``pred.npy``）。所以那条命令不会为查询窗口产出任何新文件。
#:
#: 这里用 runpy 执行官方 ``run.py`` 自己的 ``--is_training 1 --do_predict`` 分支，只把
#: ``train``/``test`` 置空，使它只做 ``predict(setting, load=True)``——权重全部来自训练阶段
#: 的官方 checkpoint，预测由官方 ``predict()`` 与官方 ``Dataset_Pred`` 完成，不改官方源码。
#: 必须原地替换方法，不能把模块全局的官方类名换成子类：官方 ``__init__``
#: 使用 ``super(Exp_Long_Term_Forecast, self)``，类名被换后会递归调用自己。
#: 最后把官方产物另存到调用方指定的窗口专属路径，不再用 glob 去找。
ITRANSFORMER_PREDICT_SNIPPET = (
    'import sys, runpy; import numpy as np; '
    'out=sys.argv.pop(1); '
    'import experiments.exp_long_term_forecasting as M; Base=M.Exp_Long_Term_Forecast; '
    'official_predict=Base.predict; '
    'Base.train=lambda self, setting: self.model; '
    'Base.test=lambda self, *a, **k: None; '
    'Base.predict=lambda self, setting, load=False, _o=out: ('
    'official_predict(self, setting, True),'
    'np.save(_o, np.load("./results/"+setting+"/real_prediction.npy")))[0]; '
    'runpy.run_path("run.py", run_name="__main__")'
)


def itransformer_predict_command(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int,
    root_path: Path,
    output: Path,
) -> List[str]:
    """官方查询入口：加载训练 checkpoint，对当前窗口 CSV 预测，写到 ``output``。"""
    return [
        str(config["python"]), "-c", ITRANSFORMER_PREDICT_SNIPPET, str(output),
    ] + itransformer_flags(
        config, model_id=model_id, data_path=data_path, root_path=root_path,
        forecast_steps=forecast_steps, do_predict=True,
    )


MOLE_TRAIN_SNIPPET = """
import os, runpy
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.data_loader import Dataset_Custom


def fixed_read_data(self):
    self.scaler = StandardScaler()
    files = ('train.csv', 'val.csv', 'test.csv')
    train = pd.read_csv(os.path.join(self.root_path, files[0]))
    split = pd.read_csv(os.path.join(self.root_path, files[self.set_type]))
    if self.features in ('M', 'MS'):
        train_values = train.drop(columns=['date'])
        split_values = split.drop(columns=['date'])
    else:
        train_values = train[[self.target]]
        split_values = split[[self.target]]
    if self.scale:
        self.scaler.fit(train_values.values)
        data = self.scaler.transform(split_values.values)
    else:
        data = split_values.values
    stamp = pd.DatetimeIndex(pd.to_datetime(split['date']))
    if self.timeenc == 0:
        data_stamp = pd.DataFrame({
            'month': stamp.month, 'day': stamp.day,
            'weekday': stamp.weekday, 'hour': stamp.hour,
        }).values
    else:
        data_stamp = time_features(stamp, freq=self.freq).transpose(1, 0)
    self.data_x = np.asarray(data, dtype=float)
    self.data_y = self.data_x
    self.data_stamp = data_stamp


Dataset_Custom.__read_data__ = fixed_read_data
runpy.run_path('run_longExp.py', run_name='__main__')
"""


def mole_train_command(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int,
    seed: int, root_path: Path,
) -> List[str]:
    """官方 `run_longExp.py` 训练。其 `--do_predict` 有真实缺陷，预测另走官方 Python API。"""
    hp = hyperparameters(config, "mole")
    return [
        str(config["python"]), "-u", "-c", MOLE_TRAIN_SNIPPET,
        "--is_training", "1", "--model_id", model_id, "--model", "MoLE_DLinear",
        "--data", "custom", "--root_path", f"{root_path}/", "--data_path", data_path,
        "--features", "S", "--target", "load", "--freq", "h",
        "--checkpoints", str(config["checkpoints"]),
        "--seq_len", str(hp["seq_len"]), "--pred_len", str(forecast_steps),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1", "--t_dim", str(hp["t_dim"]),
        "--train_epochs", str(hp["train_epochs"]), "--batch_size", str(hp["batch_size"]),
        "--patience", str(hp["patience"]), "--learning_rate", str(hp["learning_rate"]),
        "--num_workers", "0", "--itr", "1", "--des", str(hp["des"]),
        "--seed", str(seed), "--gpu", str(config.get("gpu", 0)),
    ]


#: MoLE 预测：官方 `--do_predict` 抛
#: ``TypeError: Dataset_Pred.__init__() got an unexpected keyword argument config``，
#: 因此按探针实测的方式直接调用官方 ``MoLE_DLinear.Model`` 与 ``Dataset_Pred``。
MOLE_PREDICT_SNIPPET = (
    'import random,sys; import numpy as np, torch; from types import SimpleNamespace; '
    'from data_provider.data_loader import Dataset_Pred; from models.MoLE_DLinear import Model; '
    'h=int(sys.argv[1]); ckpt=sys.argv[2]; out=sys.argv[3]; data_path=sys.argv[4]; gpu=int(sys.argv[5]); '
    'seed=int(sys.argv[6]); seq=int(sys.argv[7]); t_dim=int(sys.argv[8]); '
    'random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); '
    'cfg=SimpleNamespace(t_dim=t_dim,seq_len=seq,pred_len=h,individual=0,enc_in=1,freq="h",head_dropout=0.0); '
    'ds=Dataset_Pred(cfg,root_path="./dataset/",data_path=data_path,size=[seq,seq,h],'
    'features="S",target="load",timeenc=1,freq="h"); x,_,xmark,_=ds[0]; '
    'model=Model(cfg).cuda(gpu); model.load_state_dict(torch.load(ckpt)); model.eval(); '
    'pred=model(torch.tensor(x).float().unsqueeze(0).cuda(gpu),'
    'torch.tensor(xmark).float().unsqueeze(0).cuda(gpu)).detach().cpu().numpy(); '
    'pred=ds.inverse_transform(pred.reshape(-1,1)).reshape(1,h,1); '
    'assert pred.shape==(1,h,1) and np.isfinite(pred).all(); np.save(out,pred)'
)

#: Time-MoE：官方 README 的 AutoModelForCausalLM 路径在 transformers 4.57 上失败，
#: 探针走官方 ``TimeMoeForPrediction`` + ``generate``。上下文为 cutoff 之前的最后 720 点。
TIME_MOE_PREDICT_SNIPPET = (
    'import sys; import numpy as np, pandas as pd, torch; '
    'from time_moe.models.modeling_time_moe import TimeMoeForPrediction; '
    'from time_moe.runner import setup_seed; '
    'csv_path=sys.argv[1]; cutoff=sys.argv[2]; h=int(sys.argv[3]); out=sys.argv[4]; '
    'snapshot=sys.argv[5]; device=sys.argv[6]; context=int(sys.argv[7]); seed=int(sys.argv[8]); '
    'frame=pd.read_csv(csv_path); frame["date"]=pd.to_datetime(frame["date"]); '
    'assert frame["date"].max()==pd.Timestamp(cutoff); '
    'values=torch.tensor(frame["load"].tail(context).to_numpy(),dtype=torch.float32).unsqueeze(0).to(device); '
    'mean=values.mean(dim=-1,keepdim=True); std=values.std(dim=-1,keepdim=True); '
    'normed=(values-mean)/std; setup_seed(seed); '
    'model=TimeMoeForPrediction.from_pretrained(snapshot,device_map=device,torch_dtype="auto"); '
    'model.eval(); setup_seed(seed); '
    'generated=model.generate(normed.to(model.dtype),max_new_tokens=h); '
    'pred=generated[:,-h:].float().cpu().numpy()*float(std.cpu())+float(mean.cpu()); '
    'assert pred.shape==(1,h) and np.isfinite(pred).all(); np.save(out,pred)'
)


def run_official(command: Sequence[str], *, cwd: Path, method: str) -> None:
    """在官方仓库目录里执行官方命令；失败即抛，不吞错也不改用其他算法。"""
    argv = list(command)
    environment = None
    if method in ("itransformer", "mole") and "--gpu" in argv:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(argv[argv.index("--gpu") + 1])
    completed = subprocess.run(
        argv, cwd=str(cwd), capture_output=True, text=True, env=environment
    )
    if completed.returncode != 0:
        raise ExternalAdapterError(
            f"{method}: 官方命令失败（返回码 {completed.returncode}）\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
