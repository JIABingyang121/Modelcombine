"""多预测长度契约测试的共享装置：真实 SQLite 模型库 + 真实模型产物 + 真实原始数据。

不造假对象：模型是真实拟合/构造的 sklearn 估计器并序列化落盘，数据是真实写入
磁盘的 CSV，建库通过 ``scripts.train_combinations_kg --model-library`` 真实入口
运行。方案 §3.5 要求的入口测试都建立在这份装置上。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.artifacts import save_artifact
from src.storage.model_store import ModelStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = "pjm"
BASE_HORIZON = 1

# 递归成员：只看可由 (timestamp, load) 派生的特征
DAILY_FEATURES = ["hour", "dayofweek", "lag_1", "lag_24", "lag_168"]
WEEKLY_FEATURES = ["hour", "dayofweek", "lag_24", "lag_168"]
# 需要未来外生变量：应在离线建库边界被判定为无候选资格
WEATHER_FEATURES = ["hour", "temp"]

#: 装置真正登记的候选集合。建库必须显式声明它，声明之外的类型不参与，
#: 声明之内的必须全部登记且产物存在——这正是候选完整性规则。
FIXTURE_CANDIDATES = ["catboost_reg", "lgbm_reg", "seasonal_naive", "xgboost_reg"]


def make_series(n: int, *, start: str, seed: int) -> pd.DataFrame:
    """日/周周期 + 缓慢随机游走水平。

    随机游走是关键：它让"看 lag_1"的成员在单步上明显最优，同时让所有成员的
    误差随预测距离增长，于是单点目标与完整轨迹目标会给出不同的排序。
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="h")
    t = np.arange(n)
    level = np.cumsum(rng.normal(0, 3.0, n))
    load = (
        1000.0
        + level
        + 20.0 * np.sin(2 * np.pi * t / 24)
        + 8.0 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 1.0, n)
    )
    temp = 15.0 + 8.0 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"timestamp": ts, "load": load, "temp": temp})


def write_dataset(raw_root: Path, *, rows: int) -> dict[str, pd.DataFrame]:
    """写出 train/val/test 三个连续小时级切分。"""
    (raw_root / DATASET).mkdir(parents=True, exist_ok=True)
    frames = {}
    start = pd.Timestamp("2025-01-01")
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        frame = make_series(rows, start=str(start), seed=seed)
        frame.to_csv(raw_root / DATASET / f"{split}.csv", index=False)
        frames[split] = frame
        start = frame["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    return frames


def _supervised(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """X(t) -> y(t+1) 的 h=1 训练矩阵，lag/roll 全部来自 load 自身。"""
    work = frame.copy()
    ts = pd.to_datetime(work["timestamp"])
    work["hour"] = ts.dt.hour
    work["dayofweek"] = ts.dt.dayofweek
    for lag in (1, 24, 168):
        work[f"lag_{lag}"] = work["load"].shift(lag)
    work["target"] = work["load"].shift(-1)
    work = work.dropna().reset_index(drop=True)
    return work[features], work["target"]


def _persistence_with_gain(gain: float) -> Ridge:
    """yhat(t+1) = gain * load(t)：单步最优，递归 k 步后按 gain^k 放大。

    这正是方案 §2.2 指出的病例：在单点 h=1 的 validation 上它是最佳候选，
    但连续递归 720 步会发散，必须由完整轨迹目标把它筛掉。
    """
    model = Ridge(alpha=1.0)
    model.coef_ = np.zeros(len(DAILY_FEATURES), dtype=float)
    model.coef_[DAILY_FEATURES.index("lag_1")] = float(gain)
    model.intercept_ = 0.0
    model.n_features_in_ = len(DAILY_FEATURES)
    model.feature_names_in_ = np.asarray(DAILY_FEATURES, dtype=object)
    return model


def fit_weekly(train: pd.DataFrame, *, hour_bias: float = 0.0) -> Ridge:
    """看 lag_24/lag_168 的稳定成员。``hour_bias`` 会让它的残差随 hour 线性变化，
    从而给组合器的 interaction 分支制造真实可用的日历特征信号。"""
    model = Ridge(alpha=1.0).fit(*_supervised(train, WEEKLY_FEATURES))
    if hour_bias:
        model.coef_ = model.coef_.copy()
        model.coef_[WEEKLY_FEATURES.index("hour")] += float(hour_bias)
    return model


def register_models(db: Path, artifacts: Path, specs) -> None:
    """把 ``(model_type, model, required_features)`` 登记进模型库（产物真实落盘）。"""
    store = ModelStore(str(db))
    store.create_schema()
    for model_type, model, features in specs:
        model_id = f"{DATASET}__h{BASE_HORIZON}__{model_type}"
        path = save_artifact(model, artifacts / f"{model_id}.pkl")
        store.add_model(
            model_id=model_id,
            model_type=model_type,
            task_type="load_forecast",
            artifact_path=str(path),
            required_features=features,
            model_params={},
            lifecycle_stage="active",
        )
    store.close()


def seed_models(db: Path, artifacts: Path, train: pd.DataFrame) -> None:
    """契约测试用的四个候选：稳定 / 递归发散 / seasonal_naive / 需要未来外生变量。"""
    register_models(db, artifacts, [
        ("catboost_reg", fit_weekly(train), WEEKLY_FEATURES),
        ("lgbm_reg", _persistence_with_gain(1.002), DAILY_FEATURES),
        ("seasonal_naive", fit_weekly(train), WEEKLY_FEATURES),
        ("xgboost_reg", Ridge(alpha=1.0).fit(*_supervised(train, WEATHER_FEATURES)),
         WEATHER_FEATURES),
    ])


def run_build(tmp_path: Path, forecast_steps: list[int], *, rows: int) -> dict:
    """通过真实脚本入口建库，返回 (report, db, raw_root, artifacts)。"""
    raw_root = tmp_path / "features"
    artifacts = tmp_path / "artifacts"
    out_root = tmp_path / "out"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(raw_root, rows=rows)
    seed_models(db, artifacts, frames["train"])

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg",
            "--model-library",
            "--datasets", DATASET,
            "--forecast-steps", *[str(s) for s in forecast_steps],
            "--candidates", *FIXTURE_CANDIDATES,
            "--raw-root", str(raw_root),
            "--out-root", str(out_root),
            "--database", str(db),
            "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((out_root / "model_library_report.json").read_text())
    return {
        "report": report, "db": db, "raw_root": raw_root,
        "artifacts": artifacts, "frames": frames, "stdout": proc.stdout,
    }


def task_of(report: dict, forecast_steps: int) -> dict:
    return next(t for t in report["tasks"] if t["forecast_steps"] == forecast_steps)


def write_history(tmp_path: Path, frame: pd.DataFrame, origin: str, name: str) -> Path:
    """用户提交的历史：截止到 origin（含），只有 timestamp 与 load。"""
    ts = pd.to_datetime(frame["timestamp"])
    history = frame.loc[ts <= pd.Timestamp(origin), ["timestamp", "load"]]
    path = tmp_path / name
    history.to_csv(path, index=False)
    return path


def write_scenario(tmp_path: Path, *, forecast_steps, name="scenario.json", region=DATASET) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({
        "task_type": "load_forecast",
        "business_domain": "power_load",
        "region": region,
        "freq": "h",
        "forecast_steps": forecast_steps,
    }), encoding="utf-8")
    return path


def run_predict(db: Path, scenario: Path, history: Path, output: Path):
    return subprocess.run(
        [
            sys.executable, "run.py", "predict",
            "--database", str(db), "--scenario", str(scenario),
            "--history", str(history), "--output", str(output),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
