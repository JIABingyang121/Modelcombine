#!/usr/bin/env python3
"""V3 共享模型池：为 H/T 窗口一次性生成唯一共享基础预测（只写 yhat）。

- 模型池（7）：``seasonal_naive, prophet, random_forest, xgboost_reg, lgbm_reg,
  catboost_reg, itransformer``（V3 冻结，见 ``experiment_definition_v3.json``）。
- 训练口径（V3 冻结）：H 窗口用各自起点前 8,760 小时滚动训练；T 窗口统一用
  T1 起点前 8,760 小时训练。全程不读取任何 T 真值。
- 非外部模型：h=1 递归轨迹，特征固定为 ``final_comparison.BASE_FEATURES``。
- itransformer：官方实现，按（数据集 × 窗口 × 长度）训练一次后预测该窗口。

产物：

- ``windows/<dataset>__<label>__<model>.csv``：逐窗口逐模型明细（可断点续跑，
  已存在且行数/有限性校验通过的窗口文件会跳过）。
- ``shared_predictions.csv``：合并后的唯一共享预测长表。
- ``shared_pool_manifest.json``：训练切片、模型参数、行数与运行信息。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_comparison import BASE_FEATURES
from scripts.train_baselines import load_pipeline_model_params
from scripts.train_combinations_kg import MODEL_LIBRARY_COUNTRY_BY_REGION
from src.models.external_adapters import (
    ExternalAdapterError,
    OFFICIAL_FIXED_SEED,
    hyperparameters,
    itransformer_command,
    itransformer_predict_command,
    read_official_output,
    run_official,
    write_itransformer_splits,
    write_official_input,
)
from src.models.registry import model_registry
from src.models.trajectory_forecast import generate_member_trajectory

POOL = (
    "seasonal_naive",
    "prophet",
    "random_forest",
    "xgboost_reg",
    "lgbm_reg",
    "catboost_reg",
    "itransformer",
)
EXTERNAL_MODELS = ("itransformer",)
TREE_MODELS = ("xgboost_reg", "lgbm_reg", "catboost_reg")
POOL_SEED = 42
TRAINING_HOURS = 8760
HISTORY_HOURS = 720
RANDOM_FOREST_PARAMS: Mapping[str, Any] = {"n_estimators": 300, "n_jobs": -1}
OUTPUT_COLUMNS = (
    "dataset",
    "window_label",
    "role",
    "model",
    "forecast_steps",
    "timestamp",
    "yhat",
)


class SharedPoolError(RuntimeError):
    """共享池生成不完整：窗口取不出、模型产不出完整轨迹、产物校验不通过等。"""


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_copy(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in ("timestamp", "load") if c not in frame.columns]
    if missing:
        raise SharedPoolError(f"{path} 缺少列: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["load"] = pd.to_numeric(frame["load"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def window_history(copy: pd.DataFrame, origin: str) -> pd.DataFrame:
    origin_ts = pd.Timestamp(origin)
    start = origin_ts - pd.Timedelta(hours=HISTORY_HOURS - 1)
    history = copy[(copy["timestamp"] >= start) & (copy["timestamp"] <= origin_ts)]
    if len(history) != HISTORY_HOURS:
        raise SharedPoolError(
            f"窗口 {origin} 的历史不是 {HISTORY_HOURS} 小时（{len(history)}）"
        )
    return history.reset_index(drop=True)


def training_frame(copy: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = copy[(copy["timestamp"] >= start) & (copy["timestamp"] < end)]
    if frame.empty:
        raise SharedPoolError(f"训练切片为空: [{start}, {end})")
    return frame.reset_index(drop=True)


def build_supervised(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """``X(t) -> y(t+1)``；``y`` 带目标时刻 DatetimeIndex（prophet 训练需要真实时间轴）。"""
    work = frame[["timestamp", "load"]].copy()
    ts = pd.to_datetime(work["timestamp"])
    work["hour"] = ts.dt.hour
    work["dayofweek"] = ts.dt.dayofweek
    for lag in (1, 24, 168):
        work[f"lag_{lag}"] = work["load"].shift(lag)
    work["roll24_mean"] = work["load"].shift(1).rolling(24).mean()
    work["target"] = work["load"].shift(-1)
    work["target_ts"] = ts.shift(-1)
    work = work.dropna().reset_index(drop=True)
    if work.empty:
        raise SharedPoolError("训练切片太短，构造不出监督样本")
    x = work[list(BASE_FEATURES)]
    y = work["target"]
    y.index = pd.DatetimeIndex(work["target_ts"])
    return x, y


def fit_pool_model(
    model_type: str,
    x: pd.DataFrame,
    y: pd.Series,
    params: Mapping[str, Any],
    seed: int,
) -> Optional[Any]:
    if model_type == "seasonal_naive":
        return None
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=seed, **RANDOM_FOREST_PARAMS)
    else:
        resolved = dict(params)
        if model_type in TREE_MODELS:
            resolved["random_state"] = seed
        model = model_registry.create(model_type, **resolved)
    model.fit(x, y)
    return model


def trajectories_for(
    model: Optional[Any],
    model_type: str,
    history: pd.DataFrame,
    horizons: Sequence[int],
    country: str,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for steps in horizons:
        out[int(steps)] = generate_member_trajectory(
            model=model,
            model_type=model_type,
            required_features=list(BASE_FEATURES),
            history=history,
            forecast_steps=int(steps),
            country=country,
        )
    return out


def run_itransformer(
    config: Mapping[str, Any],
    dataset: str,
    window_label: str,
    steps: int,
    train_slice: pd.DataFrame,
    history: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    repo = Path(config["repo"])
    hp = hyperparameters(config, "itransformer")
    model_id = f"{dataset}_{window_label}_h{steps}_s{seed}"
    split_root = repo / "dataset" / f"mc_split_{model_id}"
    write_itransformer_splits(
        train_slice,
        split_root,
        seq_len=int(hp["seq_len"]),
        pred_len=int(steps),
        batch_size=int(hp["batch_size"]),
    )
    run_official(
        itransformer_command(
            config,
            model_id=model_id,
            data_path="train.csv",
            root_path=split_root,
            forecast_steps=int(steps),
        ),
        cwd=repo,
        method="itransformer",
    )
    query_name = f"mc_pred_{model_id}.csv"
    write_official_input(
        history, split_root / query_name, pd.Timestamp(history["timestamp"].max())
    )
    output = Path(config.get("output_dir", repo / "mc_output")) / f"real_prediction_{model_id}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    run_official(
        itransformer_predict_command(
            config,
            model_id=model_id,
            data_path=query_name,
            root_path=split_root,
            forecast_steps=int(steps),
            output=output,
        ),
        cwd=repo,
        method="itransformer",
    )
    return read_official_output(output, "itransformer", int(steps))


def window_file(out_dir: Path, dataset: str, label: str, model: str) -> Path:
    return out_dir / "windows" / f"{dataset}__{label}__{model}.csv"


def _valid_window_file(path: Path, horizons: Sequence[int]) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    if list(frame.columns) != list(OUTPUT_COLUMNS):
        return False
    if not np.isfinite(frame["yhat"]).all():
        return False
    counts = frame.groupby("forecast_steps")["timestamp"].nunique().to_dict()
    return counts == {int(s): int(s) for s in horizons}


def build_window_rows(
    dataset: str,
    window: Mapping[str, Any],
    model_type: str,
    values_by_steps: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    origin = pd.Timestamp(window["forecast_origin"])
    rows: List[Dict[str, Any]] = []
    for steps, values in sorted(values_by_steps.items()):
        values = np.asarray(values, dtype=float).ravel()
        if len(values) != int(steps):
            raise SharedPoolError(
                f"{dataset}/{window['label']}/{model_type}: 轨迹长度 {len(values)} != {steps}"
            )
        if not np.isfinite(values).all():
            raise SharedPoolError(
                f"{dataset}/{window['label']}/{model_type}: 轨迹含非有限值"
            )
        targets = pd.date_range(origin + pd.Timedelta(hours=1), periods=int(steps), freq="h")
        rows.extend(
            {
                "dataset": dataset,
                "window_label": window["label"],
                "role": window["role"],
                "model": model_type,
                "forecast_steps": int(steps),
                "timestamp": ts,
                "yhat": float(value),
            }
            for ts, value in zip(targets, values)
        )
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def generate_window(
    out_dir: Path,
    dataset: str,
    window: Mapping[str, Any],
    copy: pd.DataFrame,
    pool: Sequence[str],
    params: Mapping[str, Mapping[str, Any]],
    horizons: Sequence[int],
    external: Optional[Mapping[str, Any]],
    t1_origin: pd.Timestamp,
) -> None:
    origin = pd.Timestamp(window["forecast_origin"])
    history = window_history(copy, window["forecast_origin"])
    if window["role"] == "history":
        train_start = origin - pd.Timedelta(hours=TRAINING_HOURS)
        train_end = origin
    else:
        train_start = t1_origin - pd.Timedelta(hours=TRAINING_HOURS)
        train_end = t1_origin
    train_slice = training_frame(copy, train_start, train_end)
    x, y = build_supervised(train_slice)
    country = MODEL_LIBRARY_COUNTRY_BY_REGION[dataset]
    for model_type in pool:
        path = window_file(out_dir, dataset, window["label"], model_type)
        if _valid_window_file(path, horizons):
            print(f"[v3-pool] 跳过已完成: {path.name}")
            continue
        if model_type in EXTERNAL_MODELS:
            if external is None or model_type not in external:
                raise SharedPoolError(f"{model_type} 需要 --external-config 提供路径")
            config = external[model_type]
            seed = OFFICIAL_FIXED_SEED["itransformer"]
            values = {
                int(steps): run_itransformer(
                    config, dataset, window["label"], int(steps), train_slice, history, seed
                )
                for steps in horizons
            }
        else:
            model = fit_pool_model(
                model_type, x, y, params.get(model_type, {}), POOL_SEED
            )
            values = trajectories_for(model, model_type, history, horizons, country)
        frame = build_window_rows(dataset, window, model_type, values)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"[v3-pool] 写出 {path.name}: {len(frame)} 行")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="V3 共享模型池基础预测生成")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--window-plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--pipeline-config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline.yaml"
    )
    parser.add_argument("--external-config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--windows", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args(argv)

    definition = _load_json(args.definition)
    plan = _load_json(args.window_plan)
    pool = [m for m in definition["pool"] if not args.models or m in args.models]
    unknown = [m for m in pool if m not in POOL]
    if unknown:
        raise SharedPoolError(f"未知模型: {unknown}")
    horizons = [int(s) for s in definition["horizons"]]
    params = load_pipeline_model_params(args.pipeline_config)
    external = _load_json(args.external_config) if args.external_config else None

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for entry in plan["datasets"]:
        dataset = entry["dataset"]
        if args.datasets and dataset not in args.datasets:
            continue
        copy = load_copy(args.data_root / dataset / "load.csv")
        t1_origin = pd.Timestamp(
            next(w["forecast_origin"] for w in entry["windows"] if w["role"] == "test")
        )
        for window in entry["windows"]:
            if args.windows and window["label"] not in args.windows:
                continue
            generate_window(
                args.out_dir, dataset, window, copy, pool, params, horizons,
                external, t1_origin,
            )

    window_files = sorted((args.out_dir / "windows").glob("*.csv"))
    if not window_files:
        raise SharedPoolError("没有生成任何窗口产物")
    merged = pd.concat([pd.read_csv(p) for p in window_files], ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = merged.sort_values(
        ["dataset", "role", "window_label", "model", "forecast_steps", "timestamp"]
    ).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "shared_predictions.csv", index=False)
    manifest = {
        "experiment": definition["experiment"],
        "started_at": started,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "definition": str(args.definition),
        "window_plan": str(args.window_plan),
        "data_root": str(args.data_root),
        "pool": pool,
        "horizons": horizons,
        "pool_seed": POOL_SEED,
        "official_fixed_seed": {m: OFFICIAL_FIXED_SEED[m] for m in pool if m in EXTERNAL_MODELS},
        "training": {
            "history_windows": f"rolling_{TRAINING_HOURS}h_before_origin",
            "test_windows": f"single_{TRAINING_HOURS}h_before_first_test_origin",
        },
        "rows": int(len(merged)),
        "rows_by_model": {
            str(k): int(v) for k, v in merged.groupby("model").size().items()
        },
        "rows_by_window": {
            f"{k[0]}/{k[1]}": int(v)
            for k, v in merged.groupby(["dataset", "window_label"]).size().items()
        },
    }
    (args.out_dir / "shared_pool_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[v3-pool] 合并写出 {len(merged)} 行: {args.out_dir / 'shared_predictions.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SharedPoolError, ExternalAdapterError) as exc:
        print(f"[v3-pool] 失败: {exc}", file=sys.stderr)
        sys.exit(1)
