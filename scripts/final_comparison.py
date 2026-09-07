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
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

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
    _frozen_windows,
    _library_raw_frame,
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
    return pd.DataFrame({
        "timestamp": request.target_timestamps.to_numpy(),
        "y_true": np.nan,
        "yhat": yhat,
        "method": method,
        "dataset": request.dataset,
        "test_window": request.test_window,
        "forecast_steps": request.forecast_steps,
        "seed": request.seed,
    })


def run(
    *,
    methods: Sequence[str], datasets: Sequence[str], windows: Sequence[str],
    forecast_steps: Sequence[int], seeds: Sequence[int],
    raw_root: Path, window_plan: Path, database: Path | None,
    training_cutoff: Dict[str, pd.Timestamp] | None = None,
    candidates: Sequence[str] = (),
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    fitted: Dict[tuple, Any] = {}
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
                        candidates=candidates,
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
    parser.add_argument("--candidates", nargs="+", default=[],
                        help="冻结候选池；stack_ensembles_reproduction 需要，"
                             "必须与建库时声明的一致")
    parser.add_argument("--out", type=Path, required=True, help="输出长表 CSV")
    args = parser.parse_args()

    try:
        frame = run(
            methods=args.methods, datasets=args.datasets, windows=args.windows,
            forecast_steps=args.forecast_steps, seeds=args.seeds,
            raw_root=args.raw_root, window_plan=args.window_plan, database=args.database,
            candidates=args.candidates,
        )
    except FinalComparisonError as exc:
        print(f"[final] 运行不完整，立即停止: {exc}")
        return 1

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[final] {len(frame)} 行已写出: {out}")
    for (method, steps), group in frame.groupby(["method", "forecast_steps"]):
        mae = float(np.mean(np.abs(group["yhat"] - group["y_true"])))
        print(f"[final] {method} s={steps}: MAE={mae:.4f}（{len(group)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
