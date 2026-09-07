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

#: 探针实测的固定种子。iTransformer 官方 run.py 无 CLI 种子参数，硬编码 2023。
METHOD_SEEDS: Dict[str, int] = {"itransformer": 2023, "mole": 42, "time_moe": 42}

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


def itransformer_command(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int
) -> List[str]:
    """探针实测通过的官方入口（`run.py ... --pred_len H --do_predict --inverse`）。"""
    repo = Path(config["repo"])
    return [
        str(config["python"]), "-u", "run.py",
        "--is_training", "1", "--model_id", model_id, "--model", "iTransformer",
        "--data", "custom", "--root_path", f"{repo}/dataset/", "--data_path", data_path,
        "--features", "S", "--target", "load", "--freq", "h",
        "--checkpoints", str(config["checkpoints"]),
        "--seq_len", "96", "--label_len", "48", "--pred_len", str(forecast_steps),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--d_model", "32", "--n_heads", "4", "--e_layers", "1", "--d_layers", "1",
        "--d_ff", "64", "--factor", "1", "--dropout", "0.1", "--embed", "timeF",
        "--train_epochs", "1", "--batch_size", "64", "--patience", "1",
        "--learning_rate", "0.0001", "--num_workers", "0", "--itr", "1",
        "--des", "probe", "--inverse", "--do_predict", "--gpu", str(config.get("gpu", 0)),
    ]


def mole_train_command(
    config: Mapping[str, Any], *, model_id: str, data_path: str, forecast_steps: int
) -> List[str]:
    """官方 `run_longExp.py` 训练。其 `--do_predict` 有真实缺陷，预测另走官方 Python API。"""
    repo = Path(config["repo"])
    return [
        str(config["python"]), "-u", "run_longExp.py",
        "--is_training", "1", "--model_id", model_id, "--model", "MoLE_DLinear",
        "--data", "custom", "--root_path", f"{repo}/dataset/", "--data_path", data_path,
        "--features", "S", "--target", "load", "--freq", "h",
        "--checkpoints", str(config["checkpoints"]),
        "--seq_len", "96", "--pred_len", str(forecast_steps),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1", "--t_dim", "4",
        "--train_epochs", "1", "--batch_size", "64", "--patience", "1",
        "--learning_rate", "0.0001", "--num_workers", "0", "--itr", "1",
        "--des", "probe", "--seed", str(METHOD_SEEDS["mole"]),
        "--gpu", str(config.get("gpu", 0)),
    ]


#: MoLE 预测：官方 `--do_predict` 抛
#: ``TypeError: Dataset_Pred.__init__() got an unexpected keyword argument config``，
#: 因此按探针实测的方式直接调用官方 ``MoLE_DLinear.Model`` 与 ``Dataset_Pred``。
MOLE_PREDICT_SNIPPET = (
    'import random,sys; import numpy as np, torch; from types import SimpleNamespace; '
    'from data_provider.data_loader import Dataset_Pred; from models.MoLE_DLinear import Model; '
    'h=int(sys.argv[1]); ckpt=sys.argv[2]; out=sys.argv[3]; data_path=sys.argv[4]; gpu=int(sys.argv[5]); '
    'random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); '
    'cfg=SimpleNamespace(t_dim=4,seq_len=96,pred_len=h,individual=0,enc_in=1,freq="h",head_dropout=0.0); '
    'ds=Dataset_Pred(cfg,root_path="./dataset/",data_path=data_path,size=[96,96,h],'
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
    'snapshot=sys.argv[5]; device=sys.argv[6]; context=int(sys.argv[7]); '
    'frame=pd.read_csv(csv_path); frame["date"]=pd.to_datetime(frame["date"]); '
    'assert frame["date"].max()==pd.Timestamp(cutoff); '
    'values=torch.tensor(frame["load"].tail(context).to_numpy(),dtype=torch.float32).unsqueeze(0).to(device); '
    'mean=values.mean(dim=-1,keepdim=True); std=values.std(dim=-1,keepdim=True); '
    'normed=(values-mean)/std; setup_seed(42); '
    'model=TimeMoeForPrediction.from_pretrained(snapshot,device_map=device,torch_dtype="auto"); '
    'model.eval(); setup_seed(42); '
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
