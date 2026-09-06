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

本入口不训练、不选模、不读测试窗口真实值去做任何选择；真实值只用于写出 ``y_true``。
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
    MODEL_LIBRARY_BUSINESS_DOMAIN,
    MODEL_LIBRARY_COUNTRY_BY_REGION,
    MODEL_LIBRARY_TASK_TYPE,
    _frozen_windows,
    _library_raw_frame,
)
from src.models.trajectory_forecast import generate_member_trajectory
from src.storage.model_store import SUPPORTED_FORECAST_STEPS

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


def _fit_on_history_before(request: Request) -> tuple:
    """只用该窗口输入历史之前（含）的真实数据训练，绝不触及目标区间。"""
    cutoff = pd.Timestamp(request.window["forecast_origin"])
    train = request.raw[request.raw["timestamp"] <= cutoff]
    return _supervised_matrix(train)


@register("random_forest")
def _random_forest(request: Request) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    x, y = _fit_on_history_before(request)
    model = RandomForestRegressor(
        n_estimators=300, random_state=request.seed, n_jobs=-1,
    ).fit(x, y)
    return _recursive_single_model(request, model, "random_forest")


@register("xgboost")
def _xgboost(request: Request) -> np.ndarray:
    from src.models.registry import model_registry

    x, y = _fit_on_history_before(request)
    model = model_registry.create(
        "xgboost_reg", n_estimators=300, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=request.seed, n_jobs=-1,
    )
    model.fit(x, y)
    return _recursive_single_model(request, model, "xgboost")


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
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for dataset in datasets:
        raw = _library_raw_frame(raw_root, dataset)
        for steps in forecast_steps:
            frozen = _frozen_windows(window_plan, dataset, int(steps))
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
                        help="modelcombine 使用的已冻结 SQLite 模型库")
    parser.add_argument("--out", type=Path, required=True, help="输出长表 CSV")
    args = parser.parse_args()

    try:
        frame = run(
            methods=args.methods, datasets=args.datasets, windows=args.windows,
            forecast_steps=args.forecast_steps, seeds=args.seeds,
            raw_root=args.raw_root, window_plan=args.window_plan, database=args.database,
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
