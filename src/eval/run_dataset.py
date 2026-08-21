import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.eval.base_models import train_base_models, predict_models
from src.eval.combination import combine_predictions
from src.eval.data_utils import (
    DATASET_LGBM_OVERRIDES,
    ensure_dir,
    load_split,
    _debug_print_split,
    _build_row_id_from_timestamp,
    make_aligned_stack,
    prepare_supervised,
)
from src.eval.metrics import (
    compute_naive_scale,
    evaluate,
    evaluate_slices,
    seasonal_naive,
    compute_dynamic_cost,
)
from src.selector.model_health import ModelHealthChecker, ModelHealthRecord, HealthStatus
from src.selector.model_health import ModelHealthChecker as _MHC
from src.utils.drift_monitor import DriftMonitor, DriftEvent, DRIFT_CONFIG


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no", ""}


def _parse_csv_env_list(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    if raw is None:
        return []
    items = [x.strip() for x in str(raw).split(",")]
    return [x for x in items if x]


def _load_pred_from_baseline(
    baseline_root: Path,
    dataset: str,
    horizon: int,
    model_id: str,
    split: str,
    target_ts: pd.Series,
) -> np.ndarray | None:
    pred_path = baseline_root / dataset / f"{split}_pred_h{horizon}_{model_id}.csv"
    if not pred_path.exists():
        return None
    try:
        usecols = ["row_id", "timestamp", "pred"]
        pred_df = pd.read_csv(pred_path, usecols=lambda c: c in usecols)
        if "pred" not in pred_df.columns:
            return None

        target_row_id = _build_row_id_from_timestamp(pd.Series(target_ts.values)).astype(str)
        target_df = pd.DataFrame({"row_id": target_row_id.values})

        if "row_id" in pred_df.columns:
            pred_df["row_id"] = pred_df["row_id"].astype(str)
            pred_map = pred_df[["row_id", "pred"]].drop_duplicates("row_id", keep="first")
            aligned = target_df.merge(pred_map, on="row_id", how="left")["pred"].to_numpy(dtype=float)
        elif "timestamp" in pred_df.columns:
            pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
            pred_df["row_id"] = _build_row_id_from_timestamp(pred_df["timestamp"]).astype(str)
            pred_map = pred_df[["row_id", "pred"]].drop_duplicates("row_id", keep="first")
            aligned = target_df.merge(pred_map, on="row_id", how="left")["pred"].to_numpy(dtype=float)
        else:
            return None

        if len(aligned) != len(target_ts):
            return None
        nan_ratio = float(np.mean(~np.isfinite(aligned))) if len(aligned) > 0 else 1.0
        if nan_ratio > 0.5:
            return None
        if np.any(~np.isfinite(aligned)):
            fill = float(np.nanmean(aligned)) if np.isfinite(np.nanmean(aligned)) else 0.0
            aligned = np.nan_to_num(aligned, nan=fill)
        return aligned.astype(float)
    except Exception:
        return None


def run_dataset(name: str, feature_root: Path, target_col: str, horizons: List[int], out_root: Path,
                baseline_root: Path = None, max_rows: int = None, seasonal_period: int = 24) -> Tuple[Dict, List[ModelHealthRecord], List[DriftEvent]]:
    """
    修复版 + v3 优化。

    Returns:
        (ds_res, all_health_records, all_drift_events)
    """
    ds_res = {}
    all_health_records = []
    all_drift_events = []
    health_checker = ModelHealthChecker()
    drift_monitor = DriftMonitor()
    reuse_tree_preds = bool(baseline_root is not None and _env_flag("MODELCOMBINE_REUSE_BASELINE_TREE_PREDS", True))
    train_df = load_split(feature_root, "train")
    val_df = load_split(feature_root, "val")
    test_df = load_split(feature_root, "test")
    printed_split_debug = False

    for h in horizons:
        print(f"[{name}] horizon={h}")
        train_path = feature_root / "train.csv"
        val_path = feature_root / "val.csv"
        test_path = feature_root / "test.csv"

        if "london" in name.lower() and not printed_split_debug:
            _debug_print_split(train_path, train_df)
            _debug_print_split(val_path, val_df)
            _debug_print_split(test_path, test_df)
            printed_split_debug = True

        X_train, y_train, ts_train, ctx_train = prepare_supervised(train_df, target_col, h)
        X_val, y_val, ts_val, ctx_val = prepare_supervised(val_df, target_col, h)
        X_test, y_test, ts_test, ctx_test = prepare_supervised(test_df, target_col, h)

        if max_rows and len(X_train) > max_rows:
            X_train = X_train.iloc[:max_rows]
            y_train = y_train.iloc[:max_rows]
            ts_train = ts_train.iloc[:max_rows]
            ctx_train = ctx_train.iloc[:max_rows]

        tree_models = ["xgboost_reg", "lgbm_reg", "catboost_reg"]
        _model_overrides = {}
        ds_lgbm_overrides = DATASET_LGBM_OVERRIDES.get(name)
        if ds_lgbm_overrides:
            _model_overrides["lgbm_reg"] = ds_lgbm_overrides
            print(f"    [dataset override] {name} lgbm_reg 超参覆盖: {ds_lgbm_overrides}")
        fitted = {}
        val_preds: Dict[str, np.ndarray] = {}
        test_preds: Dict[str, np.ndarray] = {}
        train_status_map: Dict[str, Dict[str, object]] = {}

        loaded_tree_models: List[str] = []
        if reuse_tree_preds and baseline_root is not None:
            for mid in tree_models:
                pred_val = _load_pred_from_baseline(
                    baseline_root=baseline_root,
                    dataset=name,
                    horizon=h,
                    model_id=mid,
                    split="val",
                    target_ts=ts_val,
                )
                pred_test = _load_pred_from_baseline(
                    baseline_root=baseline_root,
                    dataset=name,
                    horizon=h,
                    model_id=mid,
                    split="test",
                    target_ts=ts_test,
                )
                if pred_val is None or pred_test is None:
                    continue
                val_preds[mid] = pred_val
                test_preds[mid] = pred_test
                loaded_tree_models.append(mid)
                meta_path = baseline_root / name / f"model_meta_h{h}_{mid}.json"
                if meta_path.exists():
                    try:
                        with meta_path.open("r", encoding="utf-8") as mf:
                            meta_obj = json.load(mf)
                        fit_status = meta_obj.get("fit_status", {})
                        if isinstance(fit_status, dict):
                            train_status_map[mid] = fit_status
                    except Exception:
                        pass
                train_status_map.setdefault(
                    mid,
                    {
                        "fit_ok": True,
                        "model_family": "tree_model",
                        "fallback_used": False,
                        "convergence_warning_count": 0,
                        "pred_source": "baseline_artifact_reuse",
                    },
                )
            if loaded_tree_models:
                print(f"    [perf] 复用 baseline 树模型预测: {loaded_tree_models}")

        missing_tree_models = [m for m in tree_models if m not in val_preds or m not in test_preds]
        if missing_tree_models:
            print(f"    [perf] 补训练缺失树模型: {missing_tree_models}")
            fitted = train_base_models(missing_tree_models, X_train, y_train, model_overrides=_model_overrides)
            val_preds.update(predict_models(fitted, X_val))
            test_preds.update(predict_models(fitted, X_test))
            for m in missing_tree_models:
                if m in val_preds and m in test_preds:
                    train_status_map.setdefault(
                        m,
                        {
                            "fit_ok": True,
                            "model_family": "tree_model",
                            "fallback_used": False,
                            "convergence_warning_count": 0,
                            "pred_source": "retrained_in_modelcombine_eval",
                        },
                    )

        external_models = ["prophet", "arima", "power_difference", "multimodal_fusion"]
        extra_external_models = _parse_csv_env_list("MODELCOMBINE_EXTRA_EXTERNAL_MODELS")
        for ext in extra_external_models:
            if ext not in external_models:
                external_models.append(ext)
        if extra_external_models:
            print(f"    [external models] 追加加载: {extra_external_models}")
        if baseline_root is not None:
            for ext_m in external_models:
                pred_val = _load_pred_from_baseline(
                    baseline_root=baseline_root,
                    dataset=name,
                    horizon=h,
                    model_id=ext_m,
                    split="val",
                    target_ts=ts_val,
                )
                pred_test = _load_pred_from_baseline(
                    baseline_root=baseline_root,
                    dataset=name,
                    horizon=h,
                    model_id=ext_m,
                    split="test",
                    target_ts=ts_test,
                )
                if pred_val is None or pred_test is None:
                    continue
                val_preds[ext_m] = pred_val
                test_preds[ext_m] = pred_test
                meta_path = baseline_root / name / f"model_meta_h{h}_{ext_m}.json"
                if meta_path.exists():
                    try:
                        with meta_path.open("r", encoding="utf-8") as mf:
                            meta_obj = json.load(mf)
                        fit_status = meta_obj.get("fit_status", {})
                        if isinstance(fit_status, dict):
                            train_status_map[ext_m] = fit_status
                    except Exception as meta_exc:
                        train_status_map[ext_m] = {
                            "fit_ok": True,
                            "fit_status_error": str(meta_exc),
                            "model_family": ext_m,
                        }
                print(f"    已加载外部模型: {ext_m}")

        all_models = tree_models + external_models
        model_cols = [m for m in all_models if m in val_preds and m in test_preds]
        print(f"    参与组合的模型: {model_cols}")

        y_val_naive_check = seasonal_naive(y_val.values, seasonal_period)
        valid_mask_check = ~np.isnan(y_val_naive_check)
        naive_rmse_val = None
        if valid_mask_check.sum() > 10:
            naive_rmse_val = float(np.sqrt(np.mean(
                (y_val.values[valid_mask_check] - y_val_naive_check[valid_mask_check]) ** 2
            )))

        health_records = health_checker.check_all(
            dataset=name, horizon=h,
            model_cols=model_cols,
            val_preds=val_preds,
            y_val=y_val.values,
            naive_rmse=naive_rmse_val,
            train_status_map=train_status_map,
        )
        all_health_records.extend(health_records)

        health_enabled = ModelHealthChecker.get_enabled_models(health_records)
        health_scene_fit = _MHC.get_scene_fit_models(health_records)
        health_disabled = ModelHealthChecker.get_disabled_models(health_records)
        if health_disabled:
            print(f"    [P1.1 健康诊断] 下线: {health_disabled}")
        scene_limited_models = [m for m in health_enabled if m not in health_scene_fit]
        if scene_limited_models:
            print(f"    [P1.1 场景受限] 仅参与静态: {scene_limited_models}")

        val_stack = make_aligned_stack(val_preds, y_val, ts_val, model_cols, name, "val", context_features=ctx_val)
        test_stack = make_aligned_stack(test_preds, y_test, ts_test, model_cols, name, "test", context_features=ctx_test)
        y_val_aligned = val_stack["y"].values
        y_test_aligned = test_stack["y"].values
        y_val_naive_aligned = seasonal_naive(y_val_aligned, seasonal_period)
        y_test_naive_aligned = seasonal_naive(y_test_aligned, seasonal_period)

        train_preds = {m: fitted[m].predict(X_train) for m in tree_models if m in fitted}
        train_stack = make_aligned_stack(train_preds, y_train, ts_train,
                                         [m for m in tree_models if m in fitted],
                                         name, "train", context_features=ctx_train)

        naive_scale = compute_naive_scale(y_train.values, seasonal_period)

        drift_events_val = []
        drift_events_test = []
        drift_cfg = DRIFT_CONFIG.get(name, {"enabled": False})
        if drift_cfg["enabled"] and len(X_train) >= drift_cfg.get("min_samples", 500):
            ctx_feature_names = [c for c in X_train.columns]
            best_base_model = min(
                [(m, float(np.mean(np.abs(y_train.values - train_preds[m])))) for m in train_preds],
                key=lambda x: x[1], default=(None, 0)
            )[0]
            train_residuals = None
            val_residuals = None
            test_residuals = None
            if best_base_model and best_base_model in train_preds:
                train_residuals = y_train.values - train_preds[best_base_model]
            if best_base_model and best_base_model in val_preds:
                val_residuals = y_val.values - val_preds[best_base_model]
            if best_base_model and best_base_model in test_preds:
                test_residuals = y_test.values - test_preds[best_base_model]

            drift_events_val = drift_monitor.check_drift(
                dataset=name, horizon=h,
                train_features=X_train.values if hasattr(X_train, 'values') else X_train,
                test_features=X_val.values if hasattr(X_val, 'values') else X_val,
                feature_names=ctx_feature_names[:min(10, len(ctx_feature_names))],
                train_residuals=train_residuals,
                test_residuals=val_residuals,
            )
            drift_events_test = drift_monitor.check_drift(
                dataset=name, horizon=h,
                train_features=X_train.values if hasattr(X_train, 'values') else X_train,
                test_features=X_test.values if hasattr(X_test, 'values') else X_test,
                feature_names=ctx_feature_names[:min(10, len(ctx_feature_names))],
                train_residuals=train_residuals,
                test_residuals=test_residuals,
            )
            # 保持全局 drift 口径稳定：历史报告仍按 train->test 统计
            all_drift_events.extend(drift_events_test)
            if drift_events_val or drift_events_test:
                print(f"    [P3.1 漂移检测] train->val={len(drift_events_val)}，train->test={len(drift_events_test)}")
                for de in drift_events_val[:5]:
                    print(f"      [val] {de.metric_type}: {de.feature_name} = {de.value:.4f} > {de.threshold}")
                for de in drift_events_test[:5]:
                    print(f"      {de.metric_type}: {de.feature_name} = {de.value:.4f} > {de.threshold}")

        combos = combine_predictions(
            val_stack, test_stack, model_cols,
            dynamic_select=True, select_strategy="top_k", top_k=3, threshold_ratio=2.0,
            naive_scale=naive_scale, horizon=h, dataset_name=name,
            train_df=train_stack,
            health_enabled_models=health_enabled,
            health_scene_fit_models=health_scene_fit,
            drift_events=drift_events_test,
            drift_events_val=drift_events_val,
            drift_events_test=drift_events_test,
            structural_baseline_val=y_val_naive_aligned,
            structural_baseline_test=y_test_naive_aligned,
            structural_baseline_name="seasonal_naive",
        )
        print(f"    naive_scale (sp={seasonal_period}): {naive_scale:.4f}")

        res_h = {"base": {}, "combos": {}, "slices": {}, "cost": {}, "meta": {
            "seasonal_period": seasonal_period,
            "naive_scale": naive_scale,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "note_val": "kg_component 策略若提供 val_pred_oof 则使用 blocked-CV OOF 口径导出；否则为 in-sample。",
            "note_val_rl_qms": "rl_qms 的 val 是在线滚动评估（用过去真值决定未来选择）",
            "note_val_dash_tta": "dash_tta 的 val 是延迟反馈在线滚动评估（仅使用 t-delay 反馈更新）",
            "note_val_mole_router": "mole_router 的 val 指标是 in-sample（仅在 val 上训练 Router）",
        }}
        total_models = len(model_cols)

        val_mask = ~np.isnan(y_val_naive_aligned)
        test_mask = ~np.isnan(y_test_naive_aligned)
        if val_mask.sum() > 0 and test_mask.sum() > 0:
            res_h["base"]["seasonal_naive"] = {
                "val": evaluate(y_val_aligned[val_mask], y_val_naive_aligned[val_mask], naive_scale),
                "test": evaluate(y_test_aligned[test_mask], y_test_naive_aligned[test_mask], naive_scale),
            }
        y_val_naive = seasonal_naive(y_val.values, seasonal_period)
        y_test_naive = seasonal_naive(y_test.values, seasonal_period)

        for m in model_cols:
            res_h["base"][m] = {
                "val": evaluate(y_val_aligned, val_stack[m].values, naive_scale),
                "test": evaluate(y_test_aligned, test_stack[m].values, naive_scale),
            }

        for c_name, c_res in combos.items():
            if c_name == "_meta":
                res_h["combos"]["_meta"] = c_res
                continue
            if not isinstance(c_res, dict) or "val_pred" not in c_res or "test_pred" not in c_res:
                continue
            combo_entry = {
                "val": evaluate(y_val_aligned, c_res["val_pred"], naive_scale),
                "test": evaluate(y_test_aligned, c_res["test_pred"], naive_scale),
                "weights": c_res["weights"],
            }
            if "category" in c_res:
                combo_entry["category"] = c_res["category"]
            if "scope" in c_res:
                combo_entry["scope"] = c_res["scope"]
            if "legacy_alias" in c_res:
                combo_entry["legacy_alias"] = c_res["legacy_alias"]
            if "description" in c_res:
                combo_entry["description"] = c_res["description"]
            if "intercept" in c_res:
                combo_entry["intercept"] = c_res["intercept"]
            if "selected_models" in c_res:
                combo_entry["selected_models"] = c_res["selected_models"]
            if "bucket_info" in c_res:
                combo_entry["bucket_info"] = c_res["bucket_info"]
            if isinstance(c_res.get("meta"), dict) and "cost_assumption" in c_res["meta"]:
                combo_entry["cost_assumption"] = c_res["meta"]["cost_assumption"]

            actual_models = c_res.get("avg_models_used", None)
            combo_entry["cost"] = compute_dynamic_cost(c_name, c_res, total_models, actual_models)

            res_h["combos"][c_name] = combo_entry

            test_stack_with_pred = test_stack.copy()
            test_stack_with_pred["_pred"] = c_res["test_pred"]
            slices = evaluate_slices(test_stack_with_pred, "_pred", model_cols)
            if slices:
                res_h["slices"][c_name] = slices

        ds_out = out_root / name
        ensure_dir(ds_out)
        val_base_payload = {
            **{m: val_preds[m] for m in model_cols},
            "row_id": val_stack["row_id"].values,
            "y": y_val.values,
            "timestamp": ts_val.values,
        }
        test_base_payload = {
            **{m: test_preds[m] for m in model_cols},
            "row_id": test_stack["row_id"].values,
            "y": y_test.values,
            "timestamp": ts_test.values,
        }
        val_base_payload["seasonal_naive"] = y_val_naive
        test_base_payload["seasonal_naive"] = y_test_naive
        pd.DataFrame(val_base_payload).to_csv(ds_out / f"val_base_h{h}.csv", index=False)
        pd.DataFrame(test_base_payload).to_csv(ds_out / f"test_base_h{h}.csv", index=False)

        combo_df = {
            "row_id": test_stack["row_id"].values,
            "y": y_test_aligned,
            "timestamp": test_stack["timestamp"],
        }
        combo_df_val = {
            "row_id": val_stack["row_id"].values,
            "y": y_val_aligned,
            "timestamp": val_stack["timestamp"],
        }
        for c_name, c_res in combos.items():
            if not isinstance(c_res, dict) or "test_pred" not in c_res:
                continue
            combo_df[c_name] = c_res["test_pred"]
            if "val_pred" in c_res:
                combo_df_val[c_name] = c_res["val_pred"]
        pd.DataFrame(combo_df).to_csv(ds_out / f"test_combos_h{h}.csv", index=False)
        pd.DataFrame(combo_df_val).to_csv(ds_out / f"val_combos_h{h}.csv", index=False)

        # 扩展候选池导出：将组合策略预测转成 baseline 兼容格式，
        # 供 KG --extended-pool 直接加载。
        if baseline_root is not None:
            export_dir = baseline_root / name
            ensure_dir(export_dir)

            strategy_val_mode = {
                "gating_network": "in_sample",
                "soft_gating": "blocked_cv_blend",
                "scenario_bucket": "in_sample",
                "gating_network_v2": "in_sample",
                "adaptive_bucket": "in_sample",
                "scenario_similarity": "in_sample",
                "rl_qms": "online_rolling",
                "dash_tta": "online_rolling",
                "mole_router": "in_sample",
                "stacking_safe": "blocked_cv_blend",
                "dynamic_stacking": "blocked_cv_blend",
                "static_weight_safe": "blocked_cv_blend",
                "dynamic_weight": "blocked_cv_blend",
                "seasonal_naive": "deterministic",
            }

            def _export_strategy_pred(
                strategy_name: str,
                val_pred: np.ndarray,
                test_pred: np.ndarray,
                val_row_ids: np.ndarray,
                test_row_ids: np.ndarray,
            ) -> None:
                val_path = export_dir / f"val_pred_h{h}_{strategy_name}.csv"
                test_path = export_dir / f"test_pred_h{h}_{strategy_name}.csv"
                val_row_ids = pd.Series(val_row_ids).astype(str).values
                test_row_ids = pd.Series(test_row_ids).astype(str).values
                pd.DataFrame({
                    "row_id": val_row_ids,
                    "timestamp": val_stack["timestamp"].values,
                    "pred": val_pred,
                    "y": y_val_aligned,
                }).to_csv(val_path, index=False)
                pd.DataFrame({
                    "row_id": test_row_ids,
                    "timestamp": test_stack["timestamp"].values,
                    "pred": test_pred,
                    "y": y_test_aligned,
                }).to_csv(test_path, index=False)
                meta_path = export_dir / f"val_pred_h{h}_{strategy_name}.meta.json"
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "strategy": strategy_name,
                            "val_eval_mode": strategy_val_mode.get(strategy_name, "unknown"),
                            "dataset": name,
                            "horizon": h,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

            export_whitelist = {
                "gating_network",
                "soft_gating",
                "scenario_bucket",
                "gating_network_v2",
                "adaptive_bucket",
                "scenario_similarity",
                "rl_qms",
                "dash_tta",
                "mole_router",
                "stacking_safe",
                "dynamic_stacking",
                "static_weight_safe",
                "dynamic_weight",
            }
            for sname, payload in combos.items():
                if sname not in export_whitelist or not isinstance(payload, dict):
                    continue
                if "val_pred" not in payload or "test_pred" not in payload:
                    continue
                val_row_ids = (
                    val_stack["row_id"].astype(str).values
                    if "row_id" in val_stack.columns
                    else np.array([f"row_{i}" for i in range(len(y_val_aligned))], dtype=object)
                )
                test_row_ids = (
                    test_stack["row_id"].astype(str).values
                    if "row_id" in test_stack.columns
                    else np.array([f"row_{i}" for i in range(len(y_test_aligned))], dtype=object)
                )
                val_pred_to_export = payload.get("val_pred_oof", payload["val_pred"])
                if "val_pred_oof" in payload:
                    strategy_val_mode[sname] = "oof" if sname == "scenario_similarity" else "blocked_cv"
                _export_strategy_pred(
                    sname,
                    val_pred_to_export,
                    payload["test_pred"],
                    val_row_ids,
                    test_row_ids,
                )

            if val_mask.sum() > 0 and test_mask.sum() > 0:
                seasonal_val = np.full(len(y_val_aligned), np.nan)
                seasonal_test = np.full(len(y_test_aligned), np.nan)
                seasonal_val[val_mask] = y_val_naive_aligned[val_mask]
                seasonal_test[test_mask] = y_test_naive_aligned[test_mask]
                val_row_ids = (
                    val_stack["row_id"].astype(str).values
                    if "row_id" in val_stack.columns
                    else np.array([f"row_{i}" for i in range(len(y_val_aligned))], dtype=object)
                )
                test_row_ids = (
                    test_stack["row_id"].astype(str).values
                    if "row_id" in test_stack.columns
                    else np.array([f"row_{i}" for i in range(len(y_test_aligned))], dtype=object)
                )
                _export_strategy_pred("seasonal_naive", seasonal_val, seasonal_test, val_row_ids, test_row_ids)

        sw_entry = combos.get("static_weight_safe", {})
        if isinstance(sw_entry.get("weights"), dict):
            ridge_weights_path = ds_out / f"ridge_weights_h{h}.json"
            with ridge_weights_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "weights": {k: float(v) for k, v in sw_entry["weights"].items()},
                    "alpha": sw_entry.get("alpha"),
                    "decay_beta": sw_entry.get("ridge_decay_beta"),
                    "selected_models": sw_entry.get("selected_models"),
                }, f, ensure_ascii=False, indent=2)

        rl_meta = combos.get("rl_qms", {})
        weights_log = rl_meta.get("weights_log")
        chosen_models = rl_meta.get("chosen_models")
        block_meta = None
        if isinstance(rl_meta.get("meta"), dict):
            block_meta = rl_meta["meta"].get("block_meta")
        if isinstance(weights_log, np.ndarray) and len(weights_log) == len(test_stack):
            weights_df = pd.DataFrame(weights_log, columns=model_cols)
            weights_df.insert(0, "timestamp", test_stack["timestamp"].values)
            weights_df.to_csv(ds_out / f"weights_log_h{h}_rl_qms.csv", index=False)
        if isinstance(chosen_models, np.ndarray) and len(chosen_models) == len(test_stack):
            chosen_ids = chosen_models.astype(int)
            chosen_names = [model_cols[i] if 0 <= i < len(model_cols) else "unknown" for i in chosen_ids]
            chosen_df = pd.DataFrame({
                "timestamp": test_stack["timestamp"].values,
                "chosen_model": chosen_names,
                "chosen_model_id": chosen_ids,
            })
            chosen_df.to_csv(ds_out / f"chosen_models_h{h}_rl_qms.csv", index=False)
            usage = dict(pd.Series(chosen_names).value_counts(normalize=True))
            with (ds_out / f"usage_hist_h{h}_rl_qms.json").open("w", encoding="utf-8") as f:
                json.dump(usage, f, ensure_ascii=False, indent=2)
        if isinstance(block_meta, list) and block_meta:
            block_df = pd.DataFrame(block_meta)
            block_df.to_csv(ds_out / f"block_meta_h{h}_rl_qms.csv", index=False)

        dash_meta = combos.get("dash_tta", {})
        dash_weights = dash_meta.get("weights_log")
        dash_chosen = dash_meta.get("chosen_models")
        dash_meta_info = dash_meta.get("meta") if isinstance(dash_meta.get("meta"), dict) else None
        if isinstance(dash_weights, np.ndarray) and len(dash_weights) == len(test_stack):
            weights_df = pd.DataFrame(dash_weights, columns=model_cols)
            # 补充第 M+1 个结构化基线专家的权重列，使行和恢复为 1.0
            weights_df["structural_baseline"] = np.clip(1.0 - dash_weights.sum(axis=1), 0.0, 1.0)
            weights_df.insert(0, "timestamp", test_stack["timestamp"].values)
            weights_df.to_csv(ds_out / f"weights_log_h{h}_dash_tta.csv", index=False)
        if isinstance(dash_chosen, np.ndarray) and len(dash_chosen) == len(test_stack):
            chosen_ids = dash_chosen.astype(int)
            chosen_names = [model_cols[i] if 0 <= i < len(model_cols) else "unknown" for i in chosen_ids]
            chosen_df = pd.DataFrame({
                "timestamp": test_stack["timestamp"].values,
                "chosen_model": chosen_names,
                "chosen_model_id": chosen_ids,
            })
            chosen_df.to_csv(ds_out / f"chosen_models_h{h}_dash_tta.csv", index=False)
            usage = dict(pd.Series(chosen_names).value_counts(normalize=True))
            with (ds_out / f"usage_hist_h{h}_dash_tta.json").open("w", encoding="utf-8") as f:
                json.dump(usage, f, ensure_ascii=False, indent=2)
        if isinstance(dash_meta_info, dict):
            with (ds_out / f"meta_h{h}_dash_tta.json").open("w", encoding="utf-8") as f:
                json.dump(dash_meta_info, f, ensure_ascii=False, indent=2)

        mole_meta = combos.get("mole_router", {})
        mole_weights = mole_meta.get("weights_log")
        mole_chosen = mole_meta.get("chosen_models")
        mole_meta_info = mole_meta.get("meta") if isinstance(mole_meta.get("meta"), dict) else None
        if isinstance(mole_weights, np.ndarray) and len(mole_weights) == len(test_stack):
            weights_df = pd.DataFrame(mole_weights, columns=model_cols)
            weights_df.insert(0, "timestamp", test_stack["timestamp"].values)
            weights_df.to_csv(ds_out / f"weights_log_h{h}_mole_router.csv", index=False)
        if isinstance(mole_chosen, np.ndarray) and len(mole_chosen) == len(test_stack):
            chosen_ids = mole_chosen.astype(int)
            chosen_names = [model_cols[i] if 0 <= i < len(model_cols) else "unknown" for i in chosen_ids]
            chosen_df = pd.DataFrame({
                "timestamp": test_stack["timestamp"].values,
                "chosen_model": chosen_names,
                "chosen_model_id": chosen_ids,
            })
            chosen_df.to_csv(ds_out / f"chosen_models_h{h}_mole_router.csv", index=False)
            usage = dict(pd.Series(chosen_names).value_counts(normalize=True))
            with (ds_out / f"usage_hist_h{h}_mole_router.json").open("w", encoding="utf-8") as f:
                json.dump(usage, f, ensure_ascii=False, indent=2)
        if isinstance(mole_meta_info, dict):
            with (ds_out / f"meta_h{h}_mole_router.json").open("w", encoding="utf-8") as f:
                json.dump(mole_meta_info, f, ensure_ascii=False, indent=2)

        ds_res[h] = res_h
    return ds_res, all_health_records, all_drift_events
