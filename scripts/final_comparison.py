#!/usr/bin/env python3
"""最终对比实验的统一入口：一种方法、一个数据集、一个测试窗口、一个预测长度、一个种子。

所有方法共用同一个契约，输出一份长表：

```text
timestamp,y_true,yhat,method,dataset,test_window,forecast_steps,seed
```

方法内部是 Python 调用还是 subprocess 调外部官方仓库，都不影响这个契约——适配器只要
返回一条长度等于 ``forecast_steps`` 的轨迹即可。因此接外部方法不需要改这个入口。

窗口来自 Stage 0 冻结的共享起点：同一个起点同时产生 H1=24、H2=168、H3=720，三种长度
共享同一份 720 小时输入历史与同一个 ``forecast_origin``。

本入口不选模、不读测试窗口真实值去做任何选择；真实值只用于写出 ``y_true``。

§5 的训练口径：所有参数只能用 **T1 之前**的数据确定。因此需要拟合的方法在
``training_cutoff``（默认取 T1 的 ``history_start``）上**只训练一次**，T1—T3 全程复用
同一个已拟合模型——不按窗口滚动重拟合。否则 T2/T3 会用上 T1 之后的数据，与"静态组合在
T1—T3 上保持同一个组合器"和"Modelcombine 基础模型全程冻结"都不一致。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage2_quality_gate import Stage2Error, _window_slice
from scripts.train_combinations_kg import (
    MODEL_LIBRARY_BASE_HORIZON,
    _library_candidate_models,
    MODEL_LIBRARY_BUSINESS_DOMAIN,
    MODEL_LIBRARY_COUNTRY_BY_REGION,
    MODEL_LIBRARY_TASK_TYPE,
    TIMESTAMP_POLICY,
    _frozen_windows,
    _library_raw_frame,
)
from src.models.external_adapters import (
    EXTERNAL_METHODS,
    REQUIRED_LOCAL_FIELDS,
    ITRANSFORMER_PREDICT_SNIPPET,
    MOLE_PREDICT_SNIPPET,
    OFFICIAL_FIXED_SEED,
    TIME_MOE_PREDICT_SNIPPET,
    ExternalAdapterError,
    hyperparameters,
    itransformer_command,
    itransformer_predict_command,
    mole_train_command,
    read_official_output,
    run_official,
    write_official_input,
    write_itransformer_splits,
)
from scripts.library_preflight import (
    LibraryIncomplete,
    assert_library_complete,
    assert_window_plan_unchanged,
)
from src.models.stack_ensemble import K_FOLDS, MultiLayerStackEnsemble
from src.models.trajectory_forecast import (
    generate_member_trajectory,
    generate_trajectory_matrix,
)
from src.storage.model_store import ModelStore
from src.storage.model_store import SUPPORTED_FORECAST_STEPS  # noqa: F401

#: 输出长表的列顺序，所有方法一致。
OUTPUT_COLUMNS = (
    "timestamp", "y_true", "yhat",
    "method", "dataset", "test_window", "forecast_steps", "seed",
)

#: §5 公平性：Random Forest、XGBoost 与 Modelcombine 的基础模型使用同一批基础特征。
#: 只取可由 (timestamp, load) 派生的列——这也是轨迹契约允许的全部输入。
BASE_FEATURES = ("hour", "dayofweek", "lag_1", "lag_24", "lag_168", "roll24_mean")


class FinalComparisonError(RuntimeError):
    """运行不完整：窗口取不出、方法产不出完整轨迹、目标时间戳不一致等。"""


class Request:
    """一次预测请求：方法拿到的全部输入，不含测试窗口真实值。"""

    def __init__(
        self, *, dataset: str, test_window: str, forecast_steps: int, seed: int,
        history: pd.DataFrame, target_timestamps: pd.Series, raw: pd.DataFrame,
        window: Dict[str, Any], database: Path | None,
        training_cutoff: pd.Timestamp, fitted: Dict[tuple, Any],
        candidates: Sequence[str] = (),
        external: Mapping[str, Any] | None = None,
        relations: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.test_window = test_window
        self.forecast_steps = int(forecast_steps)
        self.seed = int(seed)
        self.history = history
        self.target_timestamps = target_timestamps
        self.raw = raw
        self.window = window
        self.database = database
        #: 训练数据的右边界（不含）：严格早于它的真实数据才可用于拟合
        self.training_cutoff = pd.Timestamp(training_cutoff)
        #: 跨窗口共享的已拟合模型缓存，保证 T1—T3 用的是同一个模型
        self.fitted = fitted
        #: 冻结候选池：Multi-layer Stack 与 Modelcombine 共用同一批基预测（§5）
        self.candidates = list(candidates)
        #: 外部官方实现的仓库/解释器配置，按方法名索引
        self.external = dict(external or {})
        #: Modelcombine 每个查询窗口命中的关系，跨窗口累积
        self.relations = relations if relations is not None else []
        self.country = MODEL_LIBRARY_COUNTRY_BY_REGION[dataset]


Adapter = Callable[[Request], np.ndarray]
_ADAPTERS: Dict[str, Adapter] = {}


def register(name: str) -> Callable[[Adapter], Adapter]:
    def _register(adapter: Adapter) -> Adapter:
        _ADAPTERS[name] = adapter
        return adapter
    return _register


def available_methods() -> List[str]:
    return sorted(_ADAPTERS)


# ------------------------------------------------------------------ 方法适配器
@register("modelcombine")
def _modelcombine(request: Request) -> np.ndarray:
    """走 run.py predict --history 同一条代码路径：只检索已存关系并重放。"""
    from src.pipeline.main import library_predict

    if request.database is None:
        raise FinalComparisonError("modelcombine 需要 --database 指定已冻结的模型库")
    # 中间文件写进临时目录：入口只对 --out 负责，不在仓库里留下痕迹
    with tempfile.TemporaryDirectory(prefix="final_modelcombine_") as tmp:
        workdir = Path(tmp)
        scenario_path = workdir / "scenario.json"
        scenario_path.write_text(json.dumps({
            "task_type": MODEL_LIBRARY_TASK_TYPE,
            "business_domain": MODEL_LIBRARY_BUSINESS_DOMAIN,
            "region": request.dataset,
            "freq": "h",
            "forecast_steps": request.forecast_steps,
        }), encoding="utf-8")
        history_path = workdir / "history.csv"
        request.history[["timestamp", "load"]].to_csv(history_path, index=False)
        output_path = workdir / "forecast.csv"
        trace = library_predict(
            database=str(request.database), scenario=str(scenario_path),
            features=None, history=str(history_path), output=str(output_path),
        )
        if trace["selector_invoked"] is not False:
            raise FinalComparisonError(
                f"{request.dataset} {request.test_window}: trace 的 selector_invoked 不是 false"
            )
        # 命中的"场景—数据—组合"关系要留痕，供 §6 的动态选择记录与 Piece 7 分析使用
        request.relations.append({
            "dataset": request.dataset, "test_window": request.test_window,
            "forecast_steps": request.forecast_steps,
            "scenario_id": trace["scenario_id"],
            "data_profile_id": trace["data_profile_id"],
            "data_ref": trace["data_ref"],
            "data_start_at": trace["data_start_at"],
            "data_end_at": trace["data_end_at"],
            "data_similarity": trace["data_similarity"],
            "relation_id": trace["relation_id"],
            "combination_id": trace["combination_id"],
            "model_ids": trace["model_ids"],
            "member_weights": trace["member_weights"],
            "selector_invoked": trace["selector_invoked"],
        })
        return pd.read_csv(output_path)["yhat"].to_numpy(dtype=float)


def _supervised_matrix(frame: pd.DataFrame) -> tuple:
    """X(t) -> y(t+1) 的 h=1 训练矩阵，特征全部由 (timestamp, load) 派生。"""
    work = frame[["timestamp", "load"]].copy()
    ts = pd.to_datetime(work["timestamp"])
    work["hour"] = ts.dt.hour
    work["dayofweek"] = ts.dt.dayofweek
    for lag in (1, 24, 168):
        work[f"lag_{lag}"] = work["load"].shift(lag)
    work["roll24_mean"] = work["load"].shift(1).rolling(24).mean()
    work["target"] = work["load"].shift(-1)
    work = work.dropna().reset_index(drop=True)
    return work[list(BASE_FEATURES)], work["target"]


def _recursive_single_model(request: Request, model: Any, model_type: str) -> np.ndarray:
    """用与 Modelcombine 成员完全相同的递归轨迹实现产出完整轨迹。"""
    return generate_member_trajectory(
        model=model, model_type=model_type, required_features=list(BASE_FEATURES),
        history=request.history, forecast_steps=request.forecast_steps,
        country=request.country,
    )


def _fit_once(request: Request, name: str, build: Callable[[], Any]) -> Any:
    """在 ``training_cutoff`` 之前的数据上只训练一次，T1—T3 全程复用同一个模型。"""
    key = (name, request.dataset, request.seed)
    if key not in request.fitted:
        train = request.raw[request.raw["timestamp"] < request.training_cutoff]
        x, y = _supervised_matrix(train)
        if x.empty:
            raise FinalComparisonError(
                f"{request.dataset}: training_cutoff={request.training_cutoff} 之前没有可用训练样本"
            )
        model = build()
        model.fit(x, y)
        request.fitted[key] = model
    return request.fitted[key]


@register("random_forest")
def _random_forest(request: Request) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    model = _fit_once(request, "random_forest", lambda: RandomForestRegressor(
        n_estimators=300, random_state=request.seed, n_jobs=-1,
    ))
    return _recursive_single_model(request, model, "random_forest")


@register("xgboost")
def _xgboost(request: Request) -> np.ndarray:
    from src.models.registry import model_registry

    model = _fit_once(request, "xgboost", lambda: model_registry.create(
        "xgboost_reg", n_estimators=300, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=request.seed, n_jobs=-1,
    ))
    return _recursive_single_model(request, model, "xgboost")


@register("stack_ensembles_reproduction")
def _stack_ensembles_reproduction(request: Request) -> np.ndarray:
    """Multi-layer stack ensemble 的**复现版**，不是官方实现。

    与 Modelcombine 共用同一批冻结候选作为 L1（§5）。两级时序交叉验证的 K 个验证窗口
    全部取在 ``training_cutoff`` 之前，因此不触及任何测试窗口真值。
    """
    if request.database is None or not request.candidates:
        raise FinalComparisonError(
            "stack_ensembles_reproduction 需要 --database 与 --candidates"
        )
    key = ("stack_ensembles_reproduction", request.dataset, request.forecast_steps)
    if key not in request.fitted:
        members = _frozen_members(request)
        folds, columns = _stack_validation_folds(request, members)
        request.fitted[key] = (MultiLayerStackEnsemble().fit(folds), columns)
    ensemble, columns = request.fitted[key]
    base = _trajectory_matrix(request, _frozen_members(request), request.history, columns)
    return ensemble.predict(base)


def _frozen_members(request: Request) -> Dict[str, Dict[str, Any]]:
    """冻结候选池，按 (数据集) 缓存一次，避免每个窗口都重新加载产物。"""
    key = ("_members", request.dataset)
    if key not in request.fitted:
        store = ModelStore(str(request.database))
        try:
            members, _skipped = _library_candidate_models(
                store, dataset=request.dataset,
                base_horizon=MODEL_LIBRARY_BASE_HORIZON,
                model_types=request.candidates,
            )
        finally:
            store.close()
        request.fitted[key] = members
    return request.fitted[key]


def _trajectory_matrix(
    request: Request, members: Dict[str, Dict[str, Any]],
    history: pd.DataFrame, columns: Sequence[str],
) -> np.ndarray:
    """冻结候选在给定历史上的 H 步轨迹矩阵，列顺序固定为 ``columns``。"""
    matrix, skipped = generate_trajectory_matrix(
        members=members, history=history,
        forecast_steps=request.forecast_steps, country=request.country,
    )
    missing = [c for c in columns if c not in matrix.columns]
    if missing:
        reasons = {e["model_type"]: e["reason"] for e in skipped}
        raise FinalComparisonError(
            f"{request.dataset}: 冻结候选 {missing} 在该窗口产不出轨迹（{[reasons.get(m) for m in missing]}）"
        )
    return matrix[list(columns)].to_numpy(dtype=float)


def _stack_validation_folds(
    request: Request, members: Dict[str, Dict[str, Any]]
) -> tuple:
    """§3.2 的 K 折时序交叉验证窗口，全部落在 training_cutoff 之前。

    第 k 折的验证窗口是倒数第 j=(K-k+1) 个长度 H 的区间。基预测器已冻结（本方案 §5），
    因此这里只生成它们在各折上的 H 步预测，不逐折重训——见 stack_ensemble 模块文档的
    偏差说明。合格候选由轨迹契约决定：需要未来外生变量的候选在这里就会被排除。
    """
    steps = request.forecast_steps
    usable = request.raw[request.raw["timestamp"] < request.training_cutoff]
    folds: List[tuple] = []
    columns: List[str] | None = None
    for j in range(K_FOLDS, 0, -1):
        end = len(usable) - (j - 1) * steps
        start = end - steps
        if start <= 0:
            raise FinalComparisonError(
                f"{request.dataset}: training_cutoff 之前不足以容纳 {K_FOLDS} 折 × {steps} 步"
            )
        history = usable.iloc[:start]
        target = usable.iloc[start:end]
        matrix, _skipped = generate_trajectory_matrix(
            members=members, history=history,
            forecast_steps=steps, country=request.country,
        )
        available = sorted(c for c in matrix.columns if c in members)
        if not available:
            raise FinalComparisonError(
                f"{request.dataset}: 没有任何冻结候选能在验证折上产出轨迹"
            )
        columns = available if columns is None else [c for c in columns if c in available]
        folds.append((matrix, target["load"].to_numpy(dtype=float)))
    if not columns:
        raise FinalComparisonError(f"{request.dataset}: K 折之间没有共同的合格候选")
    return [(m[list(columns)].to_numpy(dtype=float), y) for m, y in folds], columns


def _external_config(request: Request, method: str) -> Dict[str, Any]:
    config = request.external.get(method)
    # 冻结定义只带口径（超参数/commit），机器本地路径只能来自 --external-config；
    # 两者合并后 config 可能非空却没有 repo，这里一并挡掉。
    if not config or "repo" not in config:
        raise FinalComparisonError(
            f"{method} 需要 --external-config 提供 repo/python 等路径（官方实现在仓库之外）"
        )
    return dict(config)


def _write_training_file(request: Request, method: str, repo: Path, seed: int) -> str:
    """冻结训练文件：``raw.timestamp < training_cutoff``，按 dataset×长度×种子写一份。

    **每次调用都按当前 raw 与 cutoff 覆盖重写**，不因同名文件已存在就复用：官方仓库在
    实验之间是持久目录，冒烟阶段留下的同名文件会被正式实验直接拿去训练。本函数只在
    ``request.fitted`` 判定"本进程尚未训练过该 model_id"时被调用，因此一个进程内仍然
    只写一次、只训练一次。
    """
    name = f"mc_train_{request.dataset}_h{request.forecast_steps}_s{seed}.csv"
    path = repo / "dataset" / name
    train = request.raw[request.raw["timestamp"] < request.training_cutoff]
    if train.empty:
        raise FinalComparisonError(
            f"{method}: {request.dataset} 在 training_cutoff 之前没有训练数据"
        )
    # 训练文件的截止点就是它自己的最后一个时间戳，不再二次截断
    write_official_input(train, path, train["timestamp"].max())
    return name


def _write_query_file(request: Request, method: str, repo: Path) -> str:
    """当前窗口的查询历史：结束时间必须等于该窗口的 forecast_origin。"""
    name = f"mc_pred_{request.dataset}_{request.test_window}_h{request.forecast_steps}.csv"
    write_official_input(
        request.history, repo / "dataset" / name,
        pd.Timestamp(request.window["forecast_origin"]),
    )
    return name


def _checkpoint_snapshot(checkpoints: Path, model_id: str) -> Dict[Path, float]:
    """训练前记下该 model_id 名下已有的 checkpoint 及其 mtime。"""
    return {
        path: path.stat().st_mtime
        for path in Path(checkpoints).glob(f"*{model_id}_*/checkpoint.pth")
    }


def _locate_trained_checkpoint(
    checkpoints: Path, model_id: str, before: Mapping[Path, float]
) -> Path:
    """定位**本次训练刚产出**的 checkpoint。

    不能只按 model_id 匹配：官方 setting（= checkpoint 目录名）还含 seq_len、模型尺寸与
    des，冒烟的探针小模型和正式模型会同时命中同一个 model_id，排序取第一个可能静默加载
    探针模型。官方 setting 的确切格式串在本机拿不到（MoLE 源码在服务器仓库外），因此这里
    不去拼那个字符串，而是直接问一个更强的问题：**哪个 checkpoint 是刚才那条训练命令写的**。

    判据是与训练前的快照逐条比对（新出现，或 mtime 变了），不是"晚于某个时刻"：文件 mtime
    来自内核的粗粒度时钟，而 ``time.time()`` 用细粒度时钟，刚写出的文件的 mtime 可能比训练
    前取的时刻还早几毫秒。命中不唯一或一个都没有时报错，绝不回头捡旧文件。
    """
    found = [
        path for path in sorted(Path(checkpoints).glob(f"*{model_id}_*/checkpoint.pth"))
        if path not in before or path.stat().st_mtime != before[path]
    ]
    if not found:
        raise FinalComparisonError(
            f"官方训练没有为 {model_id} 写出新的 checkpoint；"
            f"{checkpoints} 下的旧产物一律不采用"
        )
    if len(found) > 1:
        raise FinalComparisonError(
            f"{model_id} 匹配到多个本次训练的 checkpoint: {[str(p) for p in found]}"
        )
    return found[0]


@register("itransformer")
def _itransformer(request: Request) -> np.ndarray:
    """官方 iTransformer（thuml）。训练一次、T1—T3 各自走一次官方 ``Exp.predict``。

    官方 run.py 无 CLI 种子参数、内部固定 fix_seed=2023，因此本方法对请求种子不敏感；
    结果表里记的是官方真实种子 2023。

    查询不走 ``--is_training 0``：官方该分支只调 ``exp.test()`` 并忽略 ``--do_predict``，
    不会为查询窗口产出 ``real_prediction.npy``。详见
    ``external_adapters.ITRANSFORMER_PREDICT_SNIPPET``。
    """
    method = "itransformer"
    config = _external_config(request, method)
    repo = Path(config["repo"])
    seed = OFFICIAL_FIXED_SEED[method]
    model_id = f"{request.dataset}_h{request.forecast_steps}_s{seed}"
    hp = hyperparameters(config, method)
    # 官方 custom_fixed 从这个目录读固定的 train/val/test.csv；查询 CSV 也放这里，
    # 训练与查询共用同一个 root_path
    split_root = repo / "dataset" / f"mc_split_{request.dataset}_h{request.forecast_steps}_s{seed}"
    try:
        if ("_trained", method, model_id) not in request.fitted:
            train = request.raw[request.raw["timestamp"] < request.training_cutoff]
            if train.empty:
                raise FinalComparisonError(
                    f"{method}: {request.dataset} 在 training_cutoff 之前没有训练数据"
                )
            plan = write_itransformer_splits(
                train, split_root,
                seq_len=int(hp["seq_len"]), pred_len=request.forecast_steps,
            )
            print(f"[final] itransformer 切分 {request.dataset} h={request.forecast_steps}: {plan}")
            run_official(
                itransformer_command(
                    config, model_id=model_id, data_path="train.csv",
                    root_path=split_root, forecast_steps=request.forecast_steps,
                ),
                cwd=repo, method=method,
            )
            request.fitted[("_trained", method, model_id)] = True
        query_name = (
            f"mc_pred_{request.dataset}_{request.test_window}_h{request.forecast_steps}.csv"
        )
        write_official_input(
            request.history, split_root / query_name,
            pd.Timestamp(request.window["forecast_origin"]),
        )
        output = _external_output_path(config, repo, request, seed)
        run_official(
            itransformer_predict_command(
                config, model_id=model_id, data_path=query_name,
                root_path=split_root,
                forecast_steps=request.forecast_steps, output=output,
            ),
            cwd=repo, method=method,
        )
        return read_official_output(output, method, request.forecast_steps)
    except ExternalAdapterError as exc:
        raise FinalComparisonError(str(exc)) from exc


@register("mole")
def _mole(request: Request) -> np.ndarray:
    """官方 MoLE（rogerni）。训练一次、T1—T3 复用同一个 checkpoint。

    官方 `--do_predict` 有真实缺陷，预测按探针实测方式调用官方模型与数据类；请求的种子
    同时传给训练命令与预测调用。
    """
    method = "mole"
    config = _external_config(request, method)
    repo = Path(config["repo"])
    seed = request.seed
    model_id = f"{request.dataset}_h{request.forecast_steps}_s{seed}"
    try:
        hp = hyperparameters(config, method)
        trained = ("_trained", method, model_id)
        if trained not in request.fitted:
            train_file = _write_training_file(request, method, repo, seed)
            before = _checkpoint_snapshot(Path(config["checkpoints"]), model_id)
            run_official(
                mole_train_command(
                    config, model_id=model_id, data_path=train_file,
                    forecast_steps=request.forecast_steps, seed=seed,
                ),
                cwd=repo, method=method,
            )
            # 解析一次并记住：T1—T3 复用的必须是同一个 checkpoint，不再逐窗口重新查找
            request.fitted[trained] = _locate_trained_checkpoint(
                Path(config["checkpoints"]), model_id, before
            )
        query_file = _write_query_file(request, method, repo)
        output = _external_output_path(config, repo, request, seed)
        run_official(
            [
                str(config["python"]), "-c", MOLE_PREDICT_SNIPPET,
                str(request.forecast_steps), str(request.fitted[trained]),
                str(output), query_file, str(config.get("gpu", 0)),
                str(seed), str(hp["seq_len"]), str(hp["t_dim"]),
            ],
            cwd=repo, method=method,
        )
        return read_official_output(output, method, request.forecast_steps)
    except ExternalAdapterError as exc:
        raise FinalComparisonError(str(exc)) from exc


@register("time_moe")
def _time_moe(request: Request) -> np.ndarray:
    """官方 Time-MoE 零样本推理：不做任务训练，直接用当前窗口的历史作上下文。

    **限制**：只能证明任务输入上下文截止于该窗口的预测起点；公开预训练 checkpoint
    无法证明其预训练数据早于本实验的 training_cutoff。该限制由
    ``external_adapters.METHOD_LIMITATIONS`` 带进实验定义与结论文件。
    """
    method = "time_moe"
    config = _external_config(request, method)
    repo = Path(config["repo"])
    try:
        query_file = _write_query_file(request, method, repo)
        output = _external_output_path(config, repo, request, request.seed)
        run_official(
            [
                str(config["python"]), "-c", TIME_MOE_PREDICT_SNIPPET,
                str(repo / "dataset" / query_file),
                str(pd.Timestamp(request.window["forecast_origin"])),
                str(request.forecast_steps), str(output), str(config["snapshot"]),
                str(config.get("device", "cuda:0")), str(config.get("context_length", 720)),
                str(request.seed),
            ],
            cwd=repo, method=method,
        )
        return read_official_output(output, method, request.forecast_steps)
    except ExternalAdapterError as exc:
        raise FinalComparisonError(str(exc)) from exc


def _external_output_path(
    config: Mapping[str, Any], repo: Path, request: Request, seed: int
) -> Path:
    output_dir = Path(config.get("output_dir", repo / "mc_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{request.dataset}_{request.test_window}_h{request.forecast_steps}_s{seed}"
    path = output_dir / f"real_prediction_{tag}.npy"
    # 官方仓库是跨批次持久目录。先删掉同名旧产物：官方这次若没写出新文件，
    # read_official_output 必须失败，而不是把上一批的结果当成这次的预测读回来。
    if path.exists():
        path.unlink()
    return path


def apply_frozen_external(
    external: Mapping[str, Any] | None, frozen: Mapping[str, Any]
) -> Dict[str, Any]:
    """把冻结定义里的外部方法口径合并进运行时配置。

    ``--external-config`` 只提供机器本地路径（repo/python/checkpoints/snapshot/gpu）；
    超参数、官方代码版本、Time-MoE 权重标识一律以 ``experiment_definition.json`` 为准。
    配置里若同时写了这些字段且与冻结值不同，直接失败——定义冻结之后不允许改口径重跑。
    """
    merged = {method: dict(config) for method, config in (external or {}).items()}
    for method, values in frozen.items():
        target = merged.setdefault(method, {})
        for key, frozen_value in values.items():
            if key in target and target[key] != frozen_value:
                raise FinalComparisonError(
                    f"{method}.{key} 与冻结定义不一致：配置是 {target[key]}，"
                    f"冻结值是 {frozen_value}；定义冻结后不得改口径重跑"
                )
            target[key] = frozen_value
    return merged


def assert_seed_grid(
    methods: Sequence[str], seeds: Sequence[int], definition: Mapping[str, Any]
) -> None:
    """种子网格必须与冻结定义一致，且一次运行只能包含种子数相同的方法。

    ``run()`` 把同一组 ``--seeds`` 用于全部方法。把 MoLE（三种子）和确定性方法混在一次里
    跑，确定性方法会被重复执行三次、产出三组重复行——要到分析阶段才失败，GPU 时间已经烧完。
    所以在任何训练开始之前就拦住。
    """
    frozen = definition.get("seeds", {})
    missing = [m for m in methods if m not in frozen]
    if missing:
        raise FinalComparisonError(f"冻结定义里没有这些方法的种子: {missing}")
    counts = {m: len(frozen[m]) for m in methods}
    if len(set(counts.values())) != 1:
        raise FinalComparisonError(
            f"这些方法要求的种子数量不同 {counts}；一次运行只能用同一组 --seeds，请分批执行"
        )
    expected = next(iter(counts.values()))
    if len(set(seeds)) != len(seeds) or len(seeds) != expected:
        raise FinalComparisonError(
            f"--seeds {list(seeds)} 与冻结定义要求的 {expected} 个种子不符"
        )
    for method in methods:
        # 官方内部固定种子的方法，结果表记官方值，请求值只决定跑几次，不必相等
        if method in OFFICIAL_FIXED_SEED:
            continue
        if {int(s) for s in seeds} != {int(s) for s in frozen[method]}:
            raise FinalComparisonError(
                f"{method} 的冻结种子是 {sorted(frozen[method])}，"
                f"--seeds 是 {sorted(seeds)}"
            )


def repo_state() -> tuple:
    """主仓库的 (HEAD, 受跟踪文件改动行)。

    HEAD 相同不等于跑的是那份代码——受跟踪源码被改过就不是冻结的那个版本。未跟踪文件是
    运行产物，不参与判定，与外部仓库用的是同一套判据。
    """
    head = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if head.returncode != 0:
        raise FinalComparisonError(f"读不到主仓库 HEAD：{head.stderr.strip()}")
    dirty = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    return head.stdout.strip(), dirty.stdout.strip().splitlines()


#: 冻结定义与运行参数必须逐项一致的字段。定义是这一批实验的唯一口径，运行时不得偏离。
def assert_matches_definition(args, definition: Mapping[str, Any]) -> None:
    """在 run() 之前把运行参数与冻结定义逐项比对。

    这一步必须在 run() 之前完成：run() 的第一个动作就是按窗口切出 T1—T3 的真值
    （`_window_slice`）。任何"跑起来才发现口径不对、改完再跑一遍"的流程，等于对着测试
    窗口迭代。定义里记了什么就比什么——只记录不比对的字段等于没冻结。
    """
    problems: List[str] = []

    def _same(name: str, got: Any, want: Any) -> None:
        if got != want:
            problems.append(f"{name}: 运行用 {got!r}，冻结定义是 {want!r}")

    _same("--datasets", list(args.datasets),
          [d["dataset"] for d in definition["datasets"]])
    _same("--windows", list(args.windows), list(definition["test_windows"]))
    _same("--forecast-steps", [int(s) for s in args.forecast_steps],
          [int(s) for s in definition["forecast_steps"]])
    _same("--raw-root", str(Path(args.raw_root).resolve()), definition["raw_root"])
    plan_path_matches = str(Path(args.window_plan).resolve()) == definition["window_plan"]
    _same("--window-plan", str(Path(args.window_plan).resolve()), definition["window_plan"])
    _same("--database",
          None if args.database is None else str(Path(args.database).resolve()),
          definition["database"])
    _same("--candidates", sorted(args.candidates), sorted(definition.get("candidates", [])))
    # 重复时刻的处理方式必须与建库/冻结完全一致，否则同一段历史会切出不同的序列
    _same("时间戳策略", TIMESTAMP_POLICY, definition.get("timestamp_policy"))

    commit, dirty = repo_state()
    _same("主仓库版本", commit, definition["repo_commit"])
    if dirty:
        problems.append(
            "主仓库受跟踪文件有未提交修改，实际运行的代码不是冻结的那个提交：\n    "
            + "\n    ".join(dirty[:20])
        )

    # 窗口计划只比路径不够：同一路径被覆盖后路径仍相同，run() 却会读到新窗口。
    # 路径本身已经不符时不必再比内容，上面那条问题已经说明了。
    if plan_path_matches:
        try:
            assert_window_plan_unchanged(args.window_plan, definition)
        except LibraryIncomplete as exc:
            problems.append(str(exc))

    # 模型库完整性必须在 run() 之前查完——run() 第一步就会切出 T1—T3 的真值
    if args.database is not None:
        try:
            record = assert_library_complete(
                args.database,
                datasets=[d["dataset"] for d in definition["datasets"]],
                forecast_steps=[int(s) for s in definition["forecast_steps"]],
                candidates=list(definition["candidates"]),
                base_horizon=MODEL_LIBRARY_BASE_HORIZON,
                timestamp_policy=TIMESTAMP_POLICY,
                library_report=(
                    Path(definition["library_report"])
                    if definition.get("library_report") else None
                ),
            )
            print(f"[final] 模型库预检通过: {record}")
        except LibraryIncomplete as exc:
            problems.append(str(exc))

    if problems:
        raise FinalComparisonError(
            "运行参数与冻结定义不一致，拒绝在 T1—T3 上运行：\n  " + "\n  ".join(problems)
        )


def verify_external_versions(
    external: Mapping[str, Any], methods: Sequence[str]
) -> Dict[str, Dict[str, str]]:
    """核对冻结的官方版本与**实际**代码/权重一致。启动时做一次，不按窗口重复。

    冻结定义里的 ``commit`` / ``checkpoint_id`` 本身只是声明；不核对的话，配置可以声称用
    的是冻结版本，实际仓库却在另一个 commit 上、实际加载的是另一个 snapshot。
    """
    observed: Dict[str, Dict[str, str]] = {}
    for method in [m for m in methods if m in EXTERNAL_METHODS]:
        config = external.get(method) or {}
        missing = [k for k in REQUIRED_LOCAL_FIELDS[method] if k not in config]
        if missing:
            raise FinalComparisonError(
                f"{method}: --external-config 缺少本机字段 {missing}；"
                "冻结定义只存实验口径，不存 repo/python 这些路径"
            )
        repo = Path(config["repo"])
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise FinalComparisonError(
                f"{method}: 读不到 {repo} 的 git HEAD，无法核对冻结 commit\n"
                f"{completed.stderr.strip()}"
            )
        head = completed.stdout.strip()
        if head != config["commit"]:
            raise FinalComparisonError(
                f"{method}: 仓库 {repo} 当前 HEAD 是 {head}，"
                f"冻结定义要求 {config['commit']}"
            )
        # HEAD 相等不等于跑的是那份代码：受跟踪源码被改过就不是冻结的那个版本。
        # 未跟踪文件是实验产物（训练/查询 CSV、mc_output 下的 npy），不参与判定。
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True,
        )
        if dirty.stdout.strip():
            raise FinalComparisonError(
                f"{method}: 仓库 {repo} 的受跟踪文件有未提交修改，实际运行的代码不是 "
                f"{head}\n{dirty.stdout.strip()[:2000]}"
            )
        observed[method] = {"repo_head": head}
        if method == "time_moe":
            revision = str(config["checkpoint_id"]).partition("@")[2]
            # HuggingFace 快照目录名就是该版本的 40 位提交号；分支名（main）会随时间指向
            # 不同权重，不构成可复现的版本标识
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise FinalComparisonError(
                    f"time_moe: checkpoint_id 必须写成 <repo_id>@<40 位提交号>，"
                    f"当前是 {config['checkpoint_id']!r}；分支名或标签不是不可变版本"
                )
            snapshot = Path(config["snapshot"])
            if not snapshot.is_dir():
                raise FinalComparisonError(f"time_moe: snapshot 目录不存在: {snapshot}")
            if snapshot.name != revision:
                raise FinalComparisonError(
                    f"time_moe: 实际 snapshot 是 {snapshot.name}，"
                    f"冻结 checkpoint_id 要求 {revision}"
                )
            observed[method]["snapshot_revision"] = snapshot.name
    return observed


# ------------------------------------------------------------------ 运行
def run_request(method: str, request: Request) -> pd.DataFrame:
    adapter = _ADAPTERS.get(method)
    if adapter is None:
        raise FinalComparisonError(
            f"未注册的方法 {method}；已注册: {available_methods()}"
        )
    yhat = np.asarray(adapter(request), dtype=float).ravel()
    if len(yhat) != request.forecast_steps:
        raise FinalComparisonError(
            f"{method} 在 {request.dataset} {request.test_window} 上返回 {len(yhat)} 个点，"
            f"与 forecast_steps={request.forecast_steps} 不一致——不静默截断或补齐"
        )
    if not np.isfinite(yhat).all():
        raise FinalComparisonError(
            f"{method} 在 {request.dataset} {request.test_window} 上产生非有限值"
        )
    # iTransformer 官方无 CLI 种子、内部固定 2023：结果表记官方真实值，不记请求值
    effective_seed = OFFICIAL_FIXED_SEED.get(method, request.seed)
    return pd.DataFrame({
        "timestamp": request.target_timestamps.to_numpy(),
        "y_true": np.nan,
        "yhat": yhat,
        "method": method,
        "dataset": request.dataset,
        "test_window": request.test_window,
        "forecast_steps": request.forecast_steps,
        "seed": effective_seed,
    })


def run(
    *,
    methods: Sequence[str], datasets: Sequence[str], windows: Sequence[str],
    forecast_steps: Sequence[int], seeds: Sequence[int],
    raw_root: Path, window_plan: Path, database: Path | None,
    training_cutoff: Dict[str, pd.Timestamp] | None = None,
    candidates: Sequence[str] = (),
    external: Mapping[str, Any] | None = None,
    relations: List[Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    fitted: Dict[tuple, Any] = {}
    relations = relations if relations is not None else []
    for dataset in datasets:
        raw = _library_raw_frame(raw_root, dataset)
        for steps in forecast_steps:
            frozen = _frozen_windows(window_plan, dataset, int(steps))
            cutoff = (training_cutoff or {}).get(dataset) or pd.Timestamp(
                frozen[windows[0]]["history_start"]
            )
            for label in windows:
                if label not in frozen:
                    raise FinalComparisonError(
                        f"{dataset} forecast_steps={steps}: 窗口计划里没有 {label}"
                    )
                try:
                    history, target = _window_slice(
                        raw, frozen[label], int(steps), label=f"{dataset} {label}"
                    )
                except Stage2Error as exc:
                    raise FinalComparisonError(str(exc)) from exc
                for seed in seeds:
                    request = Request(
                        dataset=dataset, test_window=label, forecast_steps=int(steps),
                        seed=int(seed), history=history,
                        target_timestamps=target["timestamp"], raw=raw,
                        window=frozen[label], database=database,
                        training_cutoff=cutoff, fitted=fitted,
                        candidates=candidates, external=external,
                        relations=relations,
                    )
                    for method in methods:
                        frame = run_request(method, request)
                        frame["y_true"] = target["load"].to_numpy(dtype=float)
                        rows.append(frame)
    if not rows:
        raise FinalComparisonError("没有产生任何预测行")
    return pd.concat(rows, ignore_index=True)[list(OUTPUT_COLUMNS)]


def main() -> int:
    parser = argparse.ArgumentParser(description="最终对比实验统一入口")
    parser.add_argument("--methods", nargs="+", required=True,
                        help=f"已注册方法: {available_methods()}")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", default=["T1", "T2", "T3"])
    parser.add_argument("--forecast-steps", nargs="+", type=int,
                        default=list(SUPPORTED_FORECAST_STEPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="确定性方法传单个种子；深度方法传 42 43 44")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--window-plan", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=None,
                        help="modelcombine 与 stack 复现版使用的已冻结 SQLite 模型库")
    parser.add_argument("--external-config", type=Path, default=None,
                        help="外部官方实现的 repo/python/checkpoints 等机器本地路径 JSON；"
                             "itransformer/mole/time_moe 需要")
    parser.add_argument("--definition", type=Path, required=True,
                        help="freeze_final_experiment.py 冻结的 experiment_definition.json。"
                             "正式运行必需：运行参数、主仓库版本与外部实现版本都以它为准")
    parser.add_argument("--candidates", nargs="+", default=[],
                        help="冻结候选池；stack_ensembles_reproduction 需要，"
                             "必须与建库时声明的一致")
    parser.add_argument("--out", type=Path, required=True, help="输出长表 CSV")
    args = parser.parse_args()

    external = (
        json.loads(args.external_config.read_text(encoding="utf-8"))
        if args.external_config else None
    )
    relations: List[Dict[str, Any]] = []
    try:
        definition = json.loads(args.definition.read_text(encoding="utf-8"))
        # 顺序刻意：全部口径校验都在 run() 之前，run() 第一步就会取出 T1—T3 真值
        assert_matches_definition(args, definition)
        assert_seed_grid(args.methods, args.seeds, definition)
        external = apply_frozen_external(external, definition.get("external", {}))
        versions = verify_external_versions(external, args.methods)
        for method, record in versions.items():
            print(f"[final] {method} 版本已核对: {record}")
        frame = run(
            relations=relations,
            methods=args.methods, datasets=args.datasets, windows=args.windows,
            forecast_steps=args.forecast_steps, seeds=args.seeds,
            raw_root=args.raw_root, window_plan=args.window_plan, database=args.database,
            candidates=args.candidates,
            external=external,
        )
    except FinalComparisonError as exc:
        print(f"[final] 运行不完整，立即停止: {exc}")
        return 1

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if relations:
        relation_path = out.with_name(f"{out.stem}_modelcombine_relations.json")
        relation_path.write_text(
            json.dumps(relations, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"[final] Modelcombine 命中关系已写出: {relation_path}")
    print(f"[final] {len(frame)} 行已写出: {out}")
    for (method, steps), group in frame.groupby(["method", "forecast_steps"]):
        mae = float(np.mean(np.abs(group["yhat"] - group["y_true"])))
        print(f"[final] {method} s={steps}: MAE={mae:.4f}（{len(group)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
