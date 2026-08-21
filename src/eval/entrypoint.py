import json
import os
import re
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

from src.eval import data_utils
from src.eval import run_dataset as run_dataset_mod
from src.eval.combination import (
    EXPECTED_STRATEGIES,
    EXPECTED_BASELINE_CLASSIC,
    EXPECTED_BASELINE_SOTA,
    EXPECTED_KG_COMPONENT,
)
from src.eval.strategy_naming import get_strategy_naming
from src.selector.experiment_framework import (
    StrategyCategory,
    ExperimentRecord,
    compute_unified_scores,
    check_evaluation_coverage,
)
from src.selector.model_health import HealthStatus, save_model_health
from src.utils.drift_monitor import save_drift_events, compute_drift_penalty, DRIFT_CONFIG


DEFAULT_HORIZONS: Dict[str, List[int]] = {
    "pjm": [1, 6, 24],
    "aemo_vic": [1, 6, 24],
    "aemo_nsw": [1, 6, 24],
}

CORE_COVERAGE_DATASETS: List[str] = ["pjm", "aemo_vic", "aemo_nsw"]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no", ""}


def _strategy_category(strategy_name: str) -> StrategyCategory:
    if strategy_name in EXPECTED_BASELINE_CLASSIC:
        return StrategyCategory.BASELINE_CLASSIC
    if strategy_name in EXPECTED_BASELINE_SOTA:
        return StrategyCategory.BASELINE_SOTA
    return StrategyCategory.KG_COMPONENT


def _append_skipped_records(
    records: List[ExperimentRecord],
    dataset: str,
    horizon: int,
    strategies: List[str],
    status: str,
    reason: str,
    fallback_strategy: str | None = None,
) -> None:
    for strategy in strategies:
        category = _strategy_category(strategy)
        records.append(ExperimentRecord(
            dataset=dataset,
            horizon=horizon,
            strategy_name=strategy,
            category=category,
            status=status,
            skip_reason=reason,
            fallback_strategy=fallback_strategy,
            fallback_ratio=1.0 if category == StrategyCategory.KG_COMPONENT else 0.0,
        ))


def _append_failed_records(
    records: List[ExperimentRecord],
    dataset: str,
    horizon: int,
    failed_map: Dict[str, str],
) -> None:
    for strategy, reason in failed_map.items():
        category = _strategy_category(strategy)
        records.append(ExperimentRecord(
            dataset=dataset,
            horizon=horizon,
            strategy_name=strategy,
            category=category,
            status="FAILED_RUNTIME",
            skip_reason=f"策略执行失败: {reason}",
            fallback_ratio=1.0 if category == StrategyCategory.KG_COMPONENT else 0.0,
        ))


def _load_horizons_from_pipeline(default_horizons: Dict[str, List[int]]) -> Dict[str, List[int]]:
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"
    horizons = {k: list(v) for k, v in default_horizons.items()}
    if not cfg_path.exists():
        print(f"[horizon] pipeline.yaml 不存在，使用默认 horizon: {horizons}")
        return horizons
    try:
        import yaml
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        datasets = cfg.get("datasets", {}) if isinstance(cfg, dict) else {}
        for ds, ds_cfg in datasets.items():
            if not isinstance(ds_cfg, dict):
                continue
            hs = ds_cfg.get("horizons")
            if isinstance(hs, list) and hs:
                parsed = []
                for h in hs:
                    try:
                        parsed.append(int(h))
                    except Exception:
                        continue
                if parsed:
                    horizons[ds] = sorted(set(parsed))
        return horizons
    except Exception as e:
        print(f"[horizon] 读取 pipeline.yaml 失败({e})，使用默认 horizon: {horizons}")
        return horizons


def _detect_artifact_horizons(baseline_root: Path, dataset: str) -> List[int]:
    """
    探测 baseline 目录中具备 val/test 预测产物的 horizon。
    """
    ds_dir = baseline_root / dataset
    if not ds_dir.exists():
        return []
    pat = re.compile(r"^(val|test)_pred_h(\d+)_.+\.csv$")
    val_h: Set[int] = set()
    test_h: Set[int] = set()
    for p in ds_dir.glob("*_pred_h*_*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        split = m.group(1)
        h = int(m.group(2))
        if split == "val":
            val_h.add(h)
        else:
            test_h.add(h)
    return sorted(val_h & test_h)


def _resolve_runnable_horizons(
    dataset: str,
    intended_horizons: List[int],
    baseline_root: Path,
) -> Tuple[List[int], List[int]]:
    """
    规则：配置意图 ∩ baseline 实际产物。
    若 baseline 未找到任何 horizon，则回退到配置意图（避免阻断无 baseline 的本地调试流程）。
    """
    artifact_horizons = _detect_artifact_horizons(baseline_root, dataset)
    intended = sorted(set(intended_horizons))
    if not artifact_horizons:
        print(f"[horizon] {dataset}: baseline 未探测到预测产物，使用配置 horizon {intended}")
        return intended, []
    runnable = sorted(set(intended) & set(artifact_horizons))
    skipped_by_artifact = sorted(set(intended) - set(runnable))
    print(
        f"[horizon] {dataset}: intended={intended}, artifacts={artifact_horizons}, "
        f"runnable={runnable}, skipped_by_artifact={skipped_by_artifact}"
    )
    return runnable, skipped_by_artifact


def run_all(features: str, baselines: str, out: str) -> None:
    out_root = Path(out)
    baseline_root = Path(baselines)
    data_utils.ensure_dir(out_root)

    horizons: Dict[str, List[int]] = _load_horizons_from_pipeline(DEFAULT_HORIZONS)
    targets = {
        "pjm": "load",
        "aemo_vic": "load",
        "aemo_nsw": "load",
    }

    # london_train_path block removed

    max_rows = {"pjm": None, "aemo_vic": None, "aemo_nsw": None}

    seasonal_periods = {
        "pjm": 24,
        "aemo_vic": 24,
        "aemo_nsw": 24,
    }

    all_res = {}
    all_health = {}
    all_drift = []
    all_experiment_records = []

    runnable_horizons_by_dataset: Dict[str, List[int]] = {}
    skipped_artifact_horizons_by_dataset: Dict[str, List[int]] = {}
    require_full_horizon_artifacts = _env_flag("MODELCOMBINE_REQUIRE_FULL_HORIZON_ARTIFACTS", True)

    for name, hs in horizons.items():
        feature_root = Path(features) / name
        if not feature_root.exists():
            print(f"skip {name}, features not found: {feature_root}")
            continue
        runnable_hs, skipped_hs = _resolve_runnable_horizons(name, hs, baseline_root)
        runnable_horizons_by_dataset[name] = runnable_hs
        skipped_artifact_horizons_by_dataset[name] = skipped_hs

        for skipped_h in skipped_hs:
            _append_skipped_records(
                all_experiment_records,
                dataset=name,
                horizon=skipped_h,
                strategies=EXPECTED_STRATEGIES,
                status="SKIPPED_BY_ARTIFACT",
                reason="baseline 预测产物缺失，未纳入本次评估",
            )

        if skipped_hs and require_full_horizon_artifacts:
            raise RuntimeError(
                f"{name} 存在缺失 horizon 产物: {skipped_hs}。"
                "请先补齐 baseline artifact 再运行 modelcombine_eval。"
                "如需允许部分任务，设置 MODELCOMBINE_REQUIRE_FULL_HORIZON_ARTIFACTS=0。"
            )

        if runnable_hs:
            res, health_records, drift_events = run_dataset_mod.run_dataset(
                name, feature_root, targets[name], runnable_hs, out_root,
                baseline_root=baseline_root, max_rows=max_rows.get(name),
                seasonal_period=seasonal_periods.get(name, 24),
            )
            all_res[name] = res
            all_health[name] = {h: [r for r in health_records if r.horizon == h] for h in hs}
            all_drift.extend(drift_events)
        else:
            print(f"[horizon] {name}: 无可运行 horizon，全部按 SKIPPED_BY_ARTIFACT 处理")
            all_res[name] = {}
            all_health[name] = {h: [] for h in hs}

        for h in runnable_hs:
            if h not in res:
                continue
            res_h = res[h]

            combo_meta = res_h.get("combos", {}).get("_meta", {})
            h_fallback_ratio = combo_meta.get("fallback_ratio", 0.0) if isinstance(combo_meta, dict) else 0.0

            for m, m_res in res_h.get("base", {}).items():
                if m == "seasonal_naive":
                    continue
                test_metrics = m_res.get("test", {})
                val_metrics = m_res.get("val", {})
                rec = ExperimentRecord(
                    dataset=name, horizon=h, strategy_name=m,
                    category=StrategyCategory.BASELINE_CLASSIC,
                    val_rmse=val_metrics.get("rmse", float("nan")),
                    test_rmse=test_metrics.get("rmse", float("nan")),
                    val_mae=val_metrics.get("mae", float("nan")),
                    test_mae=test_metrics.get("mae", float("nan")),
                    tail_rmse=test_metrics.get("tail_rmse", float("nan")),
                    fallback_ratio=0.0,
                )
                rec.compute_gap()
                ds_drift = [e for e in all_drift if e.dataset == name and e.horizon == h]
                rec.drift_penalty = compute_drift_penalty(
                    ds_drift,
                    val_test_gap_pct=rec.gap_percent,
                    strategy_name=m,
                )
                all_experiment_records.append(rec)

            for c_name, c_res in res_h.get("combos", {}).items():
                if c_name == "_meta":
                    continue
                if not isinstance(c_res, dict) or "test" not in c_res:
                    continue
                cat_str = c_res.get("category", "baseline_classic")
                cat_map = {
                    "baseline_classic": StrategyCategory.BASELINE_CLASSIC,
                    "baseline_sota": StrategyCategory.BASELINE_SOTA,
                    "kg_node": StrategyCategory.BASELINE_CLASSIC,
                    "kg_component": StrategyCategory.KG_COMPONENT,
                }
                cat = cat_map.get(cat_str, _strategy_category(c_name))
                test_m = c_res.get("test", {})
                val_m = c_res.get("val", {})
                rec = ExperimentRecord(
                    dataset=name, horizon=h, strategy_name=c_name,
                    category=cat,
                    val_rmse=val_m.get("rmse", float("nan")),
                    test_rmse=test_m.get("rmse", float("nan")),
                    val_mae=val_m.get("mae", float("nan")),
                    test_mae=test_m.get("mae", float("nan")),
                    tail_rmse=test_m.get("tail_rmse", float("nan")),
                    fallback_ratio=h_fallback_ratio if cat == StrategyCategory.KG_COMPONENT else 0.0,
                )
                rec.compute_gap()
                ds_drift = [e for e in all_drift if e.dataset == name and e.horizon == h]
                rec.drift_penalty = compute_drift_penalty(
                    ds_drift,
                    val_test_gap_pct=rec.gap_percent,
                    strategy_name=c_name,
                )
                all_experiment_records.append(rec)

            _existing_combo_names = set(c for c in res_h.get("combos", {}).keys() if c != "_meta")
            _fallback_name = combo_meta.get("recommended_fallback", "static_weight_safe") \
                if isinstance(combo_meta, dict) else "static_weight_safe"
            _is_skipped_complex = combo_meta.get("skipped_complex_kg_component", False) \
                if isinstance(combo_meta, dict) else False
            _failed_entries = combo_meta.get("failed_strategies") \
                if isinstance(combo_meta, dict) else None
            _allowed_kg_component = combo_meta.get("kg_component_allowed_strategies") \
                if isinstance(combo_meta, dict) else None
            if not isinstance(_allowed_kg_component, list):
                _allowed_kg_component = [] if _is_skipped_complex else list(EXPECTED_KG_COMPONENT)
            _allowed_kg_component_set = set(_allowed_kg_component)

            _failed_map: Dict[str, str] = {}
            if isinstance(_failed_entries, list):
                for item in _failed_entries:
                    if not isinstance(item, dict):
                        continue
                    s = item.get("strategy")
                    if not isinstance(s, str):
                        continue
                    err = item.get("error")
                    _failed_map[s] = str(err) if err is not None else "unknown_error"

            _policy_skipped = [
                s for s in EXPECTED_KG_COMPONENT
                if s not in _existing_combo_names and s not in _allowed_kg_component_set and s not in _failed_map
            ]
            if _policy_skipped:
                _append_skipped_records(
                    all_experiment_records,
                    dataset=name,
                    horizon=h,
                    strategies=_policy_skipped,
                    status="SKIPPED_BY_POLICY",
                    reason=f"策略门控降级，回退到 {_fallback_name}",
                    fallback_strategy=_fallback_name,
                )

            _runtime_failed = {
                s: _failed_map[s]
                for s in EXPECTED_STRATEGIES
                if s in _failed_map and s not in _existing_combo_names
            }
            if _runtime_failed:
                _append_failed_records(
                    all_experiment_records,
                    dataset=name,
                    horizon=h,
                    failed_map=_runtime_failed,
                )

    all_experiment_records.sort(key=lambda r: (r.dataset, r.horizon))
    for _, group in groupby(all_experiment_records, key=lambda r: (r.dataset, r.horizon)):
        group_list = list(group)
        compute_unified_scores(group_list)

    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with (out_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"saved modelcombine metrics to {out_root / 'metrics.json'}")

    save_model_health(all_health, out_root / "model_health.json")
    print(f"saved model health to {out_root / 'model_health.json'}")

    if all_drift:
        save_drift_events(all_drift, out_root / "drift_events.json")
        print(f"saved {len(all_drift)} drift events to {out_root / 'drift_events.json'}")

    strict_core_tasks = os.environ.get("MODELCOMBINE_STRICT_CORE_TASKS", "1").strip() not in {"0", "false", "False"}
    if strict_core_tasks:
        expected_datasets = [
            ds for ds in CORE_COVERAGE_DATASETS
            if ds in horizons and (Path(features) / ds).exists()
        ]
    else:
        expected_datasets = [k for k in horizons.keys() if (Path(features) / k).exists()]
    expected_horizons = {k: horizons.get(k, []) for k in expected_datasets}
    expected_task_count = sum(len(expected_horizons.get(ds, [])) for ds in expected_datasets)
    print(
        f"[coverage] datasets={expected_datasets}, expected_task_count={expected_task_count}, "
        f"strict_core_tasks={strict_core_tasks}"
    )
    if strict_core_tasks and expected_task_count != 9:
        raise RuntimeError(
            f"核心任务矩阵异常: 期望 9 个 (dataset,horizon) 任务，当前为 {expected_task_count}。"
            f"请检查 pipeline horizon 或特征目录完整性。"
        )

    coverage = check_evaluation_coverage(
        all_experiment_records,
        expected_datasets=expected_datasets,
        expected_horizons=expected_horizons,
        expected_strategies=EXPECTED_STRATEGIES,
    )
    with (out_root / "coverage_check.json").open("w", encoding="utf-8") as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)
    cov_status = coverage["status"]
    cov_rate = coverage["coverage_rate"] * 100
    print(f"coverage check: {cov_status} ({cov_rate:.1f}%)")

    if cov_status == "fail":
        missing = coverage.get("missing", [])
        n_missing = len(missing)
        missing_triples = [(m["dataset"], m["horizon"], m["strategy"]) for m in missing]
        print(f"  [FATAL] {n_missing} 个 MISSING_RESULTS: {missing_triples[:5]}...")
        raise RuntimeError(
            f"覆盖率检查失败: {n_missing} 项结果缺失（MISSING_RESULTS）。"
            f"SKIPPED_BY_POLICY / SKIPPED_BY_ARTIFACT 已在记录中豁免，"
            f"其余缺失视为异常。详见 {out_root / 'coverage_check.json'}"
        )

    leaderboard = []
    for r in all_experiment_records:
        naming = get_strategy_naming(
            r.strategy_name,
            default_category=r.category.value if hasattr(r.category, "value") else "unknown",
        )
        leaderboard.append({
            "dataset": r.dataset,
            "horizon": r.horizon,
            "strategy": r.strategy_name,
            "strategy_display_name": naming["display_name"],
            "method_family": naming["method_family"],
            "method_route": naming["method_route"],
            "core_route": naming["core_route"],
            "category": r.category.value,
            "test_rmse": r.test_rmse,
            "test_mae": r.test_mae,
            "val_rmse": r.val_rmse,
            "gap_percent": round(r.gap_percent, 2) if not np.isnan(r.gap_percent) else None,
            "tail_rmse": r.tail_rmse if not np.isnan(r.tail_rmse) else None,
            "unified_score": r.unified_score if not np.isnan(r.unified_score) else None,
            "drift_penalty": r.drift_penalty,
            "status": r.status,
        })
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv(out_root / "leaderboard_full.csv", index=False)
    print(f"saved leaderboard to {out_root / 'leaderboard_full.csv'}")

    routing_config = {
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "updated_at": datetime.now().isoformat(),
        "intended_horizons": horizons,
        "runnable_horizons": runnable_horizons_by_dataset,
        "artifact_skipped_horizons": skipped_artifact_horizons_by_dataset,
        "strategy_naming_version": "v1",
        "min_samples_for_dynamic": data_utils.MIN_SAMPLES_FOR_DYNAMIC,
        "softgating_alpha_candidates": data_utils.SOFTGATING_ALPHA_CANDIDATES,
        "model_health_summary": {},
        "drift_config": {k: v for k, v in DRIFT_CONFIG.items()},
    }
    for ds, h_records in all_health.items():
        routing_config["model_health_summary"][ds] = {}
        for h, records in h_records.items():
            enabled = [r.model for r in records if r.status in (HealthStatus.ENABLED, HealthStatus.SCENE_LIMITED)]
            disabled = {r.model: r.reason_code for r in records if r.status == HealthStatus.DISABLED}
            routing_config["model_health_summary"][ds][str(h)] = {
                "enabled": enabled,
                "disabled": disabled,
            }
    with (out_root / "routing_config.json").open("w", encoding="utf-8") as f:
        json.dump(routing_config, f, ensure_ascii=False, indent=2)
    print(f"saved routing config to {out_root / 'routing_config.json'}")
