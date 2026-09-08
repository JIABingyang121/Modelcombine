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
REQUIRED_HYPERPARAMETERS: Dict[str, tuple] = {
    "itransformer": (
        "seq_len", "label_len", "d_model", "n_heads", "e_layers", "d_layers",
        "d_ff", "factor", "dropout", "train_epochs", "batch_size", "patience",
        "learning_rate", "des",
    ),
    "mole": (
        "seq_len", "t_dim", "train_epochs", "batch_size", "patience",
        "learning_rate", "des",
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


def itransformer_flags(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int,
    do_predict: bool,
) -> List[str]:
    """官方 ``run.py`` 的参数（不含解释器与脚本名）。

    官方的 ``setting``（也就是 checkpoint 目录名）由 model_id/model/data/features/三段
    长度/模型尺寸/des 拼成，**不含 data_path**。因此训练与查询用同一份参数、只换
    ``--data_path``，就落在同一个 checkpoint 上。
    """
    repo = Path(config["repo"])
    hp = hyperparameters(config, "itransformer")
    return [
        "--is_training", "1",
        "--model_id", model_id, "--model", "iTransformer",
        "--data", "custom", "--root_path", f"{repo}/dataset/", "--data_path", data_path,
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


def itransformer_command(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int,
) -> List[str]:
    """官方训练入口（探针实测通过的 ``--is_training 1`` 分支）。

    这里**不带** ``--do_predict``：带上它，官方会用训练文件再跑一次 predict 并写下
    ``results/<setting>/real_prediction.npy``；那份文件与任何查询窗口都无关，正是旧接线
    误当成查询结果读回的东西。训练阶段干脆不产生它。
    """
    return [str(config["python"]), "-u", "run.py"] + itransformer_flags(
        config, model_id=model_id, data_path=data_path,
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
#: 最后把官方产物另存到调用方指定的窗口专属路径，不再用 glob 去找。
ITRANSFORMER_PREDICT_SNIPPET = (
    'import sys, runpy; import numpy as np; '
    'out=sys.argv.pop(1); '
    'import experiments.exp_long_term_forecasting as M; Base=M.Exp_Long_Term_Forecast; '
    'M.Exp_Long_Term_Forecast=type("Exp_Query",(Base,),{'
    '"train":(lambda self, setting: self.model),'
    '"test":(lambda self, *a, **k: None),'
    '"predict":(lambda self, setting, load=False, _o=out: ('
    'Base.predict(self, setting, True),'
    'np.save(_o, np.load("./results/"+setting+"/real_prediction.npy")))[0])}); '
    'runpy.run_path("run.py", run_name="__main__")'
)


def itransformer_predict_command(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int,
    output: Path,
) -> List[str]:
    """官方查询入口：加载训练 checkpoint，对当前窗口 CSV 预测，写到 ``output``。"""
    return [
        str(config["python"]), "-c", ITRANSFORMER_PREDICT_SNIPPET, str(output),
    ] + itransformer_flags(
        config, model_id=model_id, data_path=data_path,
        forecast_steps=forecast_steps, do_predict=True,
    )


def mole_train_command(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int,
    seed: int,
) -> List[str]:
    """官方 `run_longExp.py` 训练。其 `--do_predict` 有真实缺陷，预测另走官方 Python API。"""
    repo = Path(config["repo"])
    hp = hyperparameters(config, "mole")
    return [
        str(config["python"]), "-u", "run_longExp.py",
        "--is_training", "1", "--model_id", model_id, "--model", "MoLE_DLinear",
        "--data", "custom", "--root_path", f"{repo}/dataset/", "--data_path", data_path,
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
    completed = subprocess.run(
        list(command), cwd=str(cwd), capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise ExternalAdapterError(
            f"{method}: 官方命令失败（返回码 {completed.returncode}）\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
