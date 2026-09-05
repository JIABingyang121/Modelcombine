"""成员模型的完整预测轨迹生成：离线建库与在线预测的唯一实现。

方案 §3.2 的契约：

```text
已训练的单个模型 + 用户历史 + forecast_steps -> 该模型的完整预测轨迹
```

关键规则（对应 §2.3 已被真实回放证伪的旧行为）：

- 每个成员各自维护一份历史，只把自己上一步的预测写回自己的 lag/rolling 特征；
  组合后的输出绝不回灌给任何成员。
- 成员按各自真实的多步语义产出轨迹，绝不把逐点调用冒充直接多步：
  ``seasonal_naive`` 重复最近 168 小时；``prophet`` 一次接收全部目标时间戳；
  ``arima`` 的 ``predict`` 只按 ``len(X)`` 从**训练序列末尾**外推、无法消费用户
  提交的最新历史，第一版直接判为无资格（逐行调用会退化成一条常数轨迹）；
  其余按行取特征的模型走独立递归。
- 特征只能由 ``(timestamp, load)`` 派生：日历特征来自未来时间戳，lag/rolling
  来自成员自己的历史。需要真实未来外生变量（天气等）的模型在这里直接判定为
  不具备当前输入契约下的候选资格，由离线建库排除，不在在线预测中临时回退。

本模块被 ``scripts/train_combinations_kg.py``（离线建库）与
``src/pipeline/library_prediction.py``（在线 ``run.py predict``）共用，避免
"离线评估一种语义、生产入口另一种语义"。
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

#: seasonal_naive 的季节周期，同时是递归成员至少需要的历史长度。
SEASONAL_PERIOD = 168
SEASONAL_NAIVE_MODEL_TYPE = "seasonal_naive"
PROPHET_MODEL_TYPE = "prophet"
ARIMA_MODEL_TYPE = "arima"

#: 由未来时间戳即可确定、无需任何未来观测的日历特征。
CALENDAR_FEATURES = ("hour", "dayofweek", "month", "is_weekend", "is_holiday")

_LAG_PATTERN = re.compile(r"^lag_(\d+)$")
_ROLL_PATTERN = re.compile(r"^roll(\d+)_(mean|std)$")


class TrajectoryForecastError(RuntimeError):
    """成员无法在当前输入契约下产生要求长度的轨迹。"""


def _requested_lags_and_windows(feature_names: Iterable[str]) -> tuple[List[int], List[int]]:
    lags: set[int] = set()
    windows: set[int] = set()
    for name in feature_names:
        lag = _LAG_PATTERN.match(name)
        if lag:
            lags.add(int(lag.group(1)))
            continue
        roll = _ROLL_PATTERN.match(name)
        if roll:
            windows.add(int(roll.group(1)))
    return sorted(lags), sorted(windows)


def underivable_features(required_features: Iterable[str]) -> List[str]:
    """返回无法由 ``(timestamp, load)`` 派生的特征名。"""
    unsupported = []
    for name in required_features:
        if name in CALENDAR_FEATURES:
            continue
        if _LAG_PATTERN.match(name) or _ROLL_PATTERN.match(name):
            continue
        unsupported.append(name)
    return unsupported


def calendar_frame(timestamps: Sequence[Any], country: str) -> pd.DataFrame:
    """由时间戳生成日历特征表；不含任何未来观测量。"""
    from scripts.generate_features import add_holiday, add_time_features

    frame = pd.DataFrame({"timestamp": pd.to_datetime(pd.Series(list(timestamps)))})
    frame = add_time_features(frame, "timestamp")
    return add_holiday(frame, "timestamp", country)


def future_timestamps(history: pd.DataFrame, forecast_steps: int) -> pd.DatetimeIndex:
    """预测起点之后的 ``forecast_steps`` 个小时级目标时间戳。"""
    origin = pd.to_datetime(history["timestamp"]).iloc[-1]
    return pd.date_range(
        origin + pd.Timedelta(hours=1), periods=int(forecast_steps), freq="h"
    )


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ("timestamp", "load") if column not in history.columns]
    if missing:
        raise TrajectoryForecastError(f"history 缺少必填列: {missing}")
    frame = history[["timestamp", "load"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["load"] = pd.to_numeric(frame["load"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def _seasonal_naive_trajectory(history: pd.DataFrame, forecast_steps: int) -> np.ndarray:
    if len(history) < SEASONAL_PERIOD:
        raise TrajectoryForecastError(
            f"seasonal_naive 需要至少 {SEASONAL_PERIOD} 小时历史，收到 {len(history)}"
        )
    season = history["load"].to_numpy(dtype=float)[-SEASONAL_PERIOD:]
    repeats = int(np.ceil(forecast_steps / SEASONAL_PERIOD))
    return np.tile(season, repeats)[:forecast_steps].astype(float)


def _prophet_trajectory(model: Any, history: pd.DataFrame, forecast_steps: int) -> np.ndarray:
    """Prophet 按目标时间戳一次输出整条轨迹。

    ``ProphetModel._make_future_frame`` 只在 ``X.index`` 是 DatetimeIndex 时使用
    调用方给的时间戳，否则从训练末尾顺延——逐行调用会让每一步都回到同一个
    ``train_end + 1``，产出常数轨迹。这里显式传入全部目标时间戳。
    """
    timestamps = future_timestamps(history, forecast_steps)
    frame = pd.DataFrame(index=pd.DatetimeIndex(timestamps))
    trajectory = np.asarray(model.predict(frame), dtype=float).ravel()
    if len(trajectory) != forecast_steps:
        raise TrajectoryForecastError(
            f"prophet 返回 {len(trajectory)} 个点，与请求的 {forecast_steps} 步不一致"
        )
    return trajectory


def _feature_row(
    history: pd.DataFrame,
    required_features: Sequence[str],
    country: str,
    lags: Sequence[int],
    windows: Sequence[int],
) -> pd.DataFrame:
    """成员自己的预测起点特征行 X(t)，用于产生 y(t+1)。

    与基础模型 h=1 的训练语义一致：日历特征取起点时刻 t，lag/rolling 取该成员
    自己历史（含它自己已产生的预测）在 t 之前的值。
    """
    from scripts.generate_features import add_lag_roll_grouped

    frame = calendar_frame(history["timestamp"], country)
    frame["load"] = history["load"].to_numpy(dtype=float)
    frame = add_lag_roll_grouped(frame, [], "timestamp", "load", list(lags), list(windows))
    row = frame.tail(1)
    missing = [name for name in required_features if name not in row.columns]
    if missing:
        raise TrajectoryForecastError(f"无法由历史派生的特征: {missing}")
    return row


def generate_member_trajectory(
    *,
    model: Any,
    model_type: str,
    required_features: Sequence[str],
    history: pd.DataFrame,
    forecast_steps: int,
    country: str,
) -> np.ndarray:
    """产生单个成员在 ``forecast_steps`` 上的完整预测轨迹。

    ``history`` 只需 ``timestamp`` 与 ``load`` 两列；返回长度严格等于
    ``forecast_steps`` 的一维数组，与 :func:`future_timestamps` 一一对应。
    """
    steps = int(forecast_steps)
    if steps <= 0:
        raise TrajectoryForecastError(f"forecast_steps 必须为正，收到 {forecast_steps}")

    own_history = _normalize_history(history)

    # ARIMA 的 predict 只用 len(X) 从训练序列末尾外推，既读不到用户历史，也无法
    # 按目标时间戳定位；逐行调用会得到一条常数轨迹。第一版直接判无资格，不为它
    # 建在线更新框架。
    if model_type == ARIMA_MODEL_TYPE:
        raise TrajectoryForecastError(
            "arima 的已训练产物无法消费用户提交的最新历史（predict 只按 len(X) "
            "从训练序列末尾外推），第一版判定其不具备当前输入契约下的候选资格"
        )

    # 以下两类各自有真实的多步输出，一次产出整条轨迹，不做逐点递归；它们都不消费
    # 特征表，因此不走 lag/rolling 可派生性检查。
    if model_type == SEASONAL_NAIVE_MODEL_TYPE:
        return _seasonal_naive_trajectory(own_history, steps)
    if model_type == PROPHET_MODEL_TYPE:
        return _prophet_trajectory(model, own_history, steps)

    unsupported = underivable_features(required_features)
    if unsupported:
        raise TrajectoryForecastError(
            f"模型 {model_type} 需要无法由 (timestamp, load) 派生的输入 {unsupported}；"
            "第一版历史数据契约不伪造未来外生变量"
        )

    if len(own_history) < SEASONAL_PERIOD:
        raise TrajectoryForecastError(
            f"递归成员 {model_type} 需要至少 {SEASONAL_PERIOD} 小时历史，"
            f"收到 {len(own_history)}"
        )

    lags, windows = _requested_lags_and_windows(required_features)
    # 只有最近 max(lag, window) + 1 行参与特征构造，避免逐步在全量历史上重算。
    tail_rows = max([*lags, *windows, 1]) + 1

    columns = list(required_features)
    trajectory = np.empty(steps, dtype=float)
    for step in range(steps):
        row = _feature_row(own_history.tail(tail_rows), columns, country, lags, windows)
        yhat = float(np.asarray(model.predict(row[columns]), dtype=float).ravel()[0])
        trajectory[step] = yhat
        # 只写回自己的历史：其他成员与组合输出都看不到这一步。
        own_history.loc[len(own_history)] = [
            own_history["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
            yhat,
        ]
    return trajectory


def generate_trajectory_matrix(
    *,
    members: Mapping[str, Mapping[str, Any]],
    history: pd.DataFrame,
    forecast_steps: int,
    country: str,
) -> tuple[pd.DataFrame, List[dict]]:
    """为多个成员各自生成轨迹，返回 ``(轨迹矩阵, 被排除候选)``。

    矩阵列为 ``target_timestamp/lead/<member columns>``（方案 §3.3.2，``y`` 由
    调用方按同一批目标时间戳补齐）。每个成员独立递归，彼此不共享历史。
    ``members`` 的键是成员列名（模型类型），值需含 ``model`` / ``model_type`` /
    ``required_features``。
    """
    timestamps = future_timestamps(history, forecast_steps)
    matrix = pd.DataFrame(
        {
            "target_timestamp": timestamps,
            "lead": np.arange(1, int(forecast_steps) + 1, dtype=int),
        }
    )
    skipped: List[dict] = []
    for name in sorted(members):
        spec = members[name]
        try:
            matrix[name] = generate_member_trajectory(
                model=spec["model"],
                model_type=spec["model_type"],
                required_features=spec["required_features"],
                history=history,
                forecast_steps=forecast_steps,
                country=country,
            )
        except TrajectoryForecastError as exc:
            skipped.append({"model_type": name, "reason": str(exc)})
    return matrix, skipped
