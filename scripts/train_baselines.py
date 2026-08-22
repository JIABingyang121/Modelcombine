import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
# [Fix] sklearn 1.4+ 废弃 squared=False，改用 root_mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # sklearn < 1.4 兼容
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

# ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import model_registry
from src.models.implementations import _XGBRegressor, _CatBoostRegressor
from src.utils.determinism import global_seed, seed_for_candidate
from src.utils.io import load_yaml


BASELINE_PROVENANCE_FILENAME = "baseline_provenance.json"
CANDIDATE_SEED_STRATEGY = "sha256(global_seed|model_id|stage)"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_row_id_from_timestamp(ts: pd.Series) -> pd.Series:
    """
    生成稳定 row_id：timestamp + 同时刻序号（字符串）。
    """
    ts_dt = pd.to_datetime(ts, errors="coerce")
    tmp = pd.DataFrame({"_ts_dt": ts_dt, "_orig_idx": np.arange(len(ts_dt), dtype=int)})
    tmp = tmp.sort_values(["_ts_dt", "_orig_idx"]).reset_index(drop=True)
    tmp["row_id"] = (
        tmp["_ts_dt"].astype(str)
        + "_"
        + tmp.groupby("_ts_dt").cumcount().astype(str)
    )
    tmp = tmp.sort_values("_orig_idx").reset_index(drop=True)
    return tmp["row_id"].astype(str)


def _extract_model_id_from_name(filename: str, prefix: str) -> str | None:
    if not filename.startswith(prefix) or not filename.endswith(".csv"):
        return None
    stem = filename[len(prefix):-4]
    return stem if stem else None


def verify_prediction_artifacts(
    out_root: Path,
    datasets: List[str],
    horizons_map: Dict[str, List[int]],
    min_models_per_task: int = 2,
) -> None:
    """
    校验 baseline 预测产物完整性。

    规则:
    1. 每个 dataset/horizon 至少有 min_models_per_task 个模型同时具备 val+test 预测。
    2. 同一 split 下不同模型的行数必须一致。
    """
    issues: List[str] = []

    for ds in datasets:
        ds_dir = out_root / ds
        if not ds_dir.exists():
            issues.append(f"{ds}: 数据集目录不存在 ({ds_dir})")
            continue

        for h in horizons_map.get(ds, []):
            val_prefix = f"val_pred_h{h}_"
            test_prefix = f"test_pred_h{h}_"

            val_files = list(ds_dir.glob(f"{val_prefix}*.csv"))
            test_files = list(ds_dir.glob(f"{test_prefix}*.csv"))

            val_models = {
                _extract_model_id_from_name(p.name, val_prefix)
                for p in val_files
            }
            test_models = {
                _extract_model_id_from_name(p.name, test_prefix)
                for p in test_files
            }
            val_models.discard(None)
            test_models.discard(None)

            common_models = sorted(val_models & test_models)
            if len(common_models) < min_models_per_task:
                issues.append(
                    f"{ds} h={h}: val/test 共同模型数不足 ({len(common_models)} < {min_models_per_task}), "
                    f"val={sorted(val_models)}, test={sorted(test_models)}"
                )
                continue

            val_row_counts = {}
            test_row_counts = {}
            for m in common_models:
                val_path = ds_dir / f"val_pred_h{h}_{m}.csv"
                test_path = ds_dir / f"test_pred_h{h}_{m}.csv"
                # 减去表头行
                with val_path.open("r", encoding="utf-8") as vf:
                    val_row_counts[m] = max(0, sum(1 for _ in vf) - 1)
                with test_path.open("r", encoding="utf-8") as tf:
                    test_row_counts[m] = max(0, sum(1 for _ in tf) - 1)

            if len(set(val_row_counts.values())) != 1:
                issues.append(f"{ds} h={h}: val 行数不一致 {val_row_counts}")
            if len(set(test_row_counts.values())) != 1:
                issues.append(f"{ds} h={h}: test 行数不一致 {test_row_counts}")

    if issues:
        issue_text = "\n  - " + "\n  - ".join(issues)
        raise RuntimeError(
            "Baseline 预测产物完整性校验失败，请先修复再进入下游流程:" + issue_text
        )


def load_split(feature_root: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(feature_root / f"{split}.csv")


def prepare_supervised(df: pd.DataFrame, target_col: str, horizon: int, time_col: str = "timestamp") -> Tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    """准备监督学习数据
    
    Args:
        df: 输入数据框
        target_col: 目标列名
        horizon: 预测步长
        time_col: 时间列名，默认 "timestamp"，支持配置
    """
    import warnings
    
    # [Fix] 检查时间列是否存在，支持多种常见时间列名
    df = df.copy()
    time_candidates = [time_col, 'timestamp', 'ts', 'datetime', 'date', 'time']
    actual_time_col = None
    for tc in time_candidates:
        if tc in df.columns:
            actual_time_col = tc
            break
    
    if actual_time_col is None:
        raise ValueError(f"数据缺少时间列，尝试了: {time_candidates}。请确保数据包含时间信息。")
    
    # [Fix] 使用 pd.to_datetime 确保正确时间排序（避免字典序）
    df[actual_time_col] = pd.to_datetime(df[actual_time_col], errors='coerce')
    
    # [Fix] 检查并警告 NaT 值
    nat_count = df[actual_time_col].isna().sum()
    if nat_count > 0:
        warnings.warn(
            f"[prepare_supervised] 时间列 '{actual_time_col}' 包含 {nat_count} 个无效时间值 (NaT)，这些行将被丢弃。"
            f"请检查数据质量。",
            UserWarning
        )
        df = df.dropna(subset=[actual_time_col])
    
    df = df.sort_values(actual_time_col)
    df[target_col] = df[target_col].astype(float)
    df["target_h"] = df[target_col].shift(-horizon)
    # keep the timestamp of the forecasted point aligned with the shifted target
    df["target_ts"] = df[actual_time_col].shift(-horizon)
    df = df.dropna(subset=["target_h", "target_ts"])  # drop rows without future target

    # [Fix] 将实际时间列也加入排除集合，防止时间列进入特征 X（潜在泄露）
    exclude = {"timestamp", actual_time_col, target_col, "target_h", "target_ts"}
    X = df.drop(columns=[c for c in exclude if c in df.columns])
    # keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    y = df["target_h"]
    ts = pd.to_datetime(df["target_ts"])
    inferred = pd.infer_freq(ts)
    if inferred is None:
        diffs = ts.diff().dropna()
        if not diffs.empty:
            median_delta = diffs.median()
            if median_delta <= pd.Timedelta(hours=1):
                inferred = "h"
            elif median_delta <= pd.Timedelta(days=1):
                inferred = "D"
    freq = inferred if inferred is not None else "h"
    
    # [Fix] 检查空序列（时间列全 NaT 或 horizon 过大）
    if len(ts) == 0:
        raise ValueError(
            f"prepare_supervised 返回空序列。可能原因：\n"
            f"  1. 时间列 '{actual_time_col}' 全部为无效值 (NaT)\n"
            f"  2. horizon={horizon} 过大，导致所有样本被 shift 后丢弃\n"
            f"  请检查数据质量和 horizon 参数。"
        )
    
    try:
        ts_index = pd.date_range(ts.iloc[0], periods=len(ts), freq=freq)
    except (OutOfBoundsDatetime, ValueError):
        ts_index = ts
    # assign time index for time-series models
    X.index = ts_index
    y.index = ts_index
    return X, y, ts, freq


def load_pipeline_model_params(pipeline_config: Path) -> Dict[str, Dict[str, Any]]:
    """读取默认入口的模型参数，供 Task 7 v4 基线训练复用。"""
    raw = load_yaml(str(pipeline_config)) or {}
    models = raw.get("models", {}) if isinstance(raw, Mapping) else {}
    if not isinstance(models, Mapping):
        raise ValueError(f"pipeline config models must be a mapping: {pipeline_config}")
    return {
        str(model_id): dict(params)
        for model_id, params in models.items()
        if isinstance(params, Mapping)
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_baseline_provenance(
    *,
    pipeline_config: Path,
    model_params: Mapping[str, Mapping[str, Any]],
    global_seed: int,
    artifacts: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """构造可供影子运行器验证的基线训练来源清单。"""
    return {
        "schema_version": 1,
        "pipeline_config": {
            "path": str(pipeline_config),
            "sha256": _sha256_file(pipeline_config),
        },
        "candidate_seed_strategy": CANDIDATE_SEED_STRATEGY,
        "global_seed": int(global_seed),
        "models": {str(model_id): dict(params) for model_id, params in model_params.items()},
        "artifacts": list(artifacts or []),
    }


def load_verified_baseline_provenance(pred_root: Path, pipeline_config: Path) -> Dict[str, Any]:
    """读取并校验基线来源清单与本次默认入口配置是否一致。"""
    path = pred_root / BASELINE_PROVENANCE_FILENAME
    if not path.exists():
        raise ValueError(f"baseline provenance is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline provenance is invalid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("baseline provenance schema_version must be 1")
    if payload.get("candidate_seed_strategy") != CANDIDATE_SEED_STRATEGY:
        raise ValueError("baseline provenance candidate_seed_strategy is not aligned")
    if payload.get("global_seed") != global_seed():
        raise ValueError("baseline provenance global_seed does not match current default")
    config = payload.get("pipeline_config")
    if not isinstance(config, dict):
        raise ValueError("baseline provenance missing pipeline_config")
    if config.get("sha256") != _sha256_file(pipeline_config):
        raise ValueError("baseline provenance pipeline_config sha256 does not match current config")
    if not isinstance(payload.get("artifacts"), list):
        raise ValueError("baseline provenance artifacts must be a list")
    return payload


def _artifact_file_names(horizon: int, model_id: str) -> Tuple[str, str, str]:
    return (
        f"val_pred_h{horizon}_{model_id}.csv",
        f"test_pred_h{horizon}_{model_id}.csv",
        f"model_meta_h{horizon}_{model_id}.json",
    )


def verify_task_artifacts(
    provenance: Mapping[str, Any],
    *,
    pred_root: Path,
    dataset: str,
    horizon: int,
    models: List[str],
) -> None:
    """确认影子矩阵实际使用的每个模型都由本轮基线清单绑定。"""
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("baseline provenance artifacts must be a list")
    index: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, Mapping):
            continue
        key = (str(record.get("dataset")), record.get("horizon"), str(record.get("model")))
        if key in index:
            raise ValueError(f"baseline provenance duplicate artifact record: {key}")
        index[key] = record
    for model_id in models:
        key = (dataset, int(horizon), str(model_id))
        record = index.get(key)
        if record is None:
            raise ValueError(f"baseline provenance missing artifact for {dataset} h={horizon} {model_id}")
        files = record.get("files")
        if not isinstance(files, Mapping):
            raise ValueError(f"baseline provenance files missing for {dataset} h={horizon} {model_id}")
        for file_name in _artifact_file_names(horizon, model_id):
            path = pred_root / dataset / file_name
            expected_hash = files.get(file_name)
            if not isinstance(expected_hash, str):
                raise ValueError(f"baseline provenance hash missing for {dataset} h={horizon} {model_id}: {file_name}")
            if not path.exists() or _sha256_file(path) != expected_hash:
                raise ValueError(f"baseline provenance artifact hash mismatch for {dataset} h={horizon} {model_id}: {file_name}")


def build_model(model_id: str, freq: str, params: Mapping[str, Any] | None = None):
    resolved = dict(params or {})
    if model_id == "arima":
        resolved.setdefault("freq", freq)
    return model_registry.create(model_id, **resolved)


def _extract_fit_status(model: object) -> Dict[str, object]:
    """
    提取模型训练状态，供下游健康诊断做收敛降权。
    """
    status: Dict[str, object] = {
        "fit_ok": True,
        "model_family": model.__class__.__name__,
        "fallback_used": False,
        "convergence_warning_count": 0,
        "warning_messages": [],
    }
    try:
        if hasattr(model, "get_fit_status") and callable(getattr(model, "get_fit_status")):
            raw = getattr(model, "get_fit_status")()
            if isinstance(raw, dict):
                status.update(raw)
        elif hasattr(model, "fit_status"):
            raw = getattr(model, "fit_status")
            if isinstance(raw, dict):
                status.update(raw)
    except Exception as exc:
        status["fit_status_error"] = str(exc)
    return status


def score_prediction(y_eval: pd.Series, pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_eval, pred)
    rmse = root_mean_squared_error(y_eval, pred)
    return {"mae": mae, "rmse": rmse}


def run_dataset(
    name: str,
    feature_root: Path,
    target_col: str,
    horizons: List[int],
    out_root: Path,
    max_rows: int | None,
    model_params: Mapping[str, Mapping[str, Any]],
) -> Dict:
    results = {}
    # 同一数据集的 split 对所有 horizon 复用，避免重复 IO。
    train_df = load_split(feature_root, "train")
    val_df = load_split(feature_root, "val")
    test_df = load_split(feature_root, "test")

    for h in horizons:
        print(f"[{name}] horizon={h}")

        X_train, y_train, _, freq = prepare_supervised(train_df, target_col, h)
        X_val, y_val, ts_val, _ = prepare_supervised(val_df, target_col, h)
        X_test, y_test, ts_test, _ = prepare_supervised(test_df, target_col, h)

        # align feature columns across splits
        train_cols = X_train.columns
        X_val = X_val.reindex(columns=train_cols, fill_value=0)
        X_test = X_test.reindex(columns=train_cols, fill_value=0)

        # Downsample large datasets for speed/memory
        cap = 200_000 if name.startswith("london") and max_rows is None else max_rows
        if cap and len(X_train) > cap:
            X_train = X_train.iloc[:cap]
            y_train = y_train.iloc[:cap]

        # 训练集合以 configs/pipeline.yaml 的 models: 段为唯一真源。
        # 此处原为硬编码 7 个模型，导致配置里新增的候选（如 seasonal_naive）
        # 永远不会被训练，而 model_params 只被用来取参数——配置与训练集合是
        # 两份互不校验的清单。本项目已在 model_assets.yaml / registry /
        # combination_utils.MODELS 上重复踩过同类漂移，故改为单一来源。
        models = list(model_params.keys())
        if not models:
            raise ValueError(
                "model_params is empty:训练集合来自 pipeline.yaml 的 models: 段，"
                "不能为空"
            )
        results_h = {"val": {}, "test": {}, "artifacts": []}
        for mid in models:
            try:
                # 每个模型每个 horizon 只训练一次，再分别在 val/test 推理。
                training_seed = seed_for_candidate(mid, stage=f"baseline:{name}:h{h}:fit")
                params = dict(model_params.get(mid, {}))
                model = build_model(mid, freq, params)
                model.fit(X_train, y_train)
                pred_val = model.predict(X_val)
                pred_test = model.predict(X_test)
                val_metrics = score_prediction(y_val, pred_val)
                test_metrics = score_prediction(y_test, pred_test)

                results_h["val"][mid] = val_metrics
                results_h["test"][mid] = test_metrics

                # save predictions
                val_path = out_root / name / f"val_pred_h{h}_{mid}.csv"
                test_path = out_root / name / f"test_pred_h{h}_{mid}.csv"
                ensure_dir(val_path.parent)
                val_row_ids = _build_row_id_from_timestamp(pd.Series(ts_val.values))
                test_row_ids = _build_row_id_from_timestamp(pd.Series(ts_test.values))
                pd.DataFrame({
                    "row_id": val_row_ids.values,
                    "timestamp": ts_val.values,
                    "pred": pred_val,
                    "y": y_val.values,
                }).to_csv(val_path, index=False)
                pd.DataFrame({
                    "row_id": test_row_ids.values,
                    "timestamp": ts_test.values,
                    "pred": pred_test,
                    "y": y_test.values,
                }).to_csv(test_path, index=False)
                fit_status = _extract_fit_status(model)
                model_meta_path = out_root / name / f"model_meta_h{h}_{mid}.json"
                with model_meta_path.open("w", encoding="utf-8") as mf:
                    json.dump(
                        {
                            "dataset": name,
                            "horizon": int(h),
                            "model": mid,
                            "model_params": params,
                            "training_seed": training_seed,
                            "candidate_seed_strategy": CANDIDATE_SEED_STRATEGY,
                            "fit_status": fit_status,
                        },
                        mf,
                        ensure_ascii=False,
                        indent=2,
                    )
                artifact_names = _artifact_file_names(h, mid)
                artifact_paths = [out_root / name / file_name for file_name in artifact_names]
                results_h["artifacts"].append(
                    {
                        "dataset": name,
                        "horizon": int(h),
                        "model": mid,
                        "files": {
                            file_name: _sha256_file(path)
                            for file_name, path in zip(artifact_names, artifact_paths)
                        },
                    }
                )

                if fit_status.get("convergence_warning_count", 0):
                    print(
                        f"  {mid}: convergence_warnings={fit_status.get('convergence_warning_count')} "
                        f"(recorded in {model_meta_path.name})"
                    )

                print(
                    f"  {mid}: VAL MAE={val_metrics['mae']:.4f} RMSE={val_metrics['rmse']:.4f} | "
                    f"TEST MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f}"
                )
            except Exception as exc:
                results_h["val"][mid] = {"error": str(exc)}
                results_h["test"][mid] = {"error": str(exc)}
                print(f"  {mid} failed: {exc}")
        results[h] = results_h
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_comparativetest1.yaml", help="配置文件路径 (未直接使用，预留)")
    parser.add_argument(
        "--pipeline-config",
        default="configs/pipeline.yaml",
        help="与默认 Protocol B 候选池对齐的模型参数配置",
    )
    parser.add_argument("--features", default="data/features", help="特征根目录")
    parser.add_argument("--out", default="reports/baselines", help="输出目录")
    parser.add_argument("--datasets", nargs="*", default=None, help="仅运行指定数据集，如 pjm aemo_vic aemo_nsw")
    parser.add_argument("--max_rows", type=int, default=None, help="训练集最大行数（全局）")
    parser.add_argument("--allow_partial", action="store_true",
                        help="允许部分 dataset/horizon 产物缺失（默认严格校验完整性）")
    args = parser.parse_args()

    dep_status = []
    if _XGBRegressor is None:
        dep_status.append("xgboost: FALLBACK (GradientBoostingRegressor)")
    if _CatBoostRegressor is None:
        dep_status.append("catboost: FALLBACK (GradientBoostingRegressor)")
    if dep_status:
        msg = "WARNING: 基线模型依赖退化:\n  " + "\n  ".join(dep_status)
        print(msg)
        if os.environ.get("MODELCOMBINE_STRICT_DEPS", "0").strip().lower() in {"1", "true", "yes"}:
            raise RuntimeError(msg)

    out_root = Path(args.out)
    ensure_dir(out_root)
    pipeline_config = Path(args.pipeline_config)
    if not pipeline_config.is_absolute():
        pipeline_config = PROJECT_ROOT / pipeline_config
    model_params = load_pipeline_model_params(pipeline_config)

    # Horizon设定与target映射
    horizons = {
        "pjm": [1, 6, 24],
        "aemo_vic": [1, 6, 24],
        "aemo_nsw": [1, 6, 24],
    }
    targets = {
        "pjm": "load",
        "aemo_vic": "load",
        "aemo_nsw": "load",
    }

    all_results = {}
    datasets = [
        ("pjm", Path(args.features) / "pjm"),
        ("aemo_vic", Path(args.features) / "aemo_vic"),
        ("aemo_nsw", Path(args.features) / "aemo_nsw"),
    ]
    if args.datasets:
        selected = set(args.datasets)
        datasets = [d for d in datasets if d[0] in selected]

    for name, root in datasets:
        if not root.exists():
            print(f"skip {name}, features not found: {root}")
            continue
        res = run_dataset(
            name,
            root,
            targets[name],
            horizons[name],
            out_root,
            args.max_rows,
            model_params,
        )
        all_results[name] = res

    if not args.allow_partial:
        selected_dataset_names = [name for name, _ in datasets]
        min_models_per_task = 2
        try:
            min_models_per_task = max(1, int(os.environ.get("MODELCOMBINE_MIN_MODELS_PER_TASK", "2")))
        except Exception:
            min_models_per_task = 2
        verify_prediction_artifacts(
            out_root,
            selected_dataset_names,
            horizons,
            min_models_per_task=min_models_per_task,
        )
        print(f"[OK] baseline artifacts integrity passed (min_models_per_task={min_models_per_task})")

    metrics_val_path = out_root / "val_metrics.json"
    metrics_test_path = out_root / "test_metrics.json"

    # split val/test metrics
    val_only = {ds: {h: res[h]["val"] for h in res} for ds, res in all_results.items()}
    test_only = {ds: {h: res[h]["test"] for h in res} for ds, res in all_results.items()}

    with metrics_val_path.open("w", encoding="utf-8") as f:
        json.dump(val_only, f, ensure_ascii=False, indent=2)
    with metrics_test_path.open("w", encoding="utf-8") as f:
        json.dump(test_only, f, ensure_ascii=False, indent=2)
    artifacts = [
        artifact
        for dataset_results in all_results.values()
        for horizon_result in dataset_results.values()
        for artifact in horizon_result.get("artifacts", [])
    ]
    provenance_path = out_root / BASELINE_PROVENANCE_FILENAME
    provenance_path.write_text(
        json.dumps(
            build_baseline_provenance(
                pipeline_config=pipeline_config,
                model_params=model_params,
                global_seed=global_seed(),
                artifacts=artifacts,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved val metrics to {metrics_val_path}")
    print(f"saved test metrics to {metrics_test_path}")
    print(f"saved baseline provenance to {provenance_path}")


if __name__ == "__main__":
    main()
