"""
统一结果对比脚本

汇总 baseline、KG Core、KG Component 三类策略，并输出口径一致性检查。
"""

import json
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.strategy_naming import get_strategy_naming

try:
    import pandas as pd
except Exception:
    pd = None

DATASET_HORIZONS = {
    "pjm": [1, 6, 24],
    "aemo_vic": [1, 6, 24],
    "aemo_nsw": [1, 6, 24],
}

CORE_METRICS = ["mae", "rmse", "smape", "mase"]
ROBUST_METRICS = ["peak_weighted_mae", "top10_mae", "top10_rmse", "tail_rmse", "p95_ae", "max_ae"]
SLICE_KEYS = [
    "period_morning_peak",
    "period_daytime",
    "period_evening_peak",
    "period_night",
    "weekday",
    "weekend",
    "holiday",
    "non_holiday",
    "top10_load",
    "vol_low",
    "vol_mid",
    "vol_high",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_format(value, fmt=".2f"):
    """安全格式化数值。"""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def safe_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "N/A"


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def build_task_id(dataset: Any, horizon: Any) -> str:
    return f"{dataset} h={horizon}"


def gap_pct(base_value: Optional[float], target_value: Optional[float]) -> Optional[float]:
    if base_value is None or target_value is None:
        return None
    try:
        base = float(base_value)
        target = float(target_value)
    except Exception:
        return None
    if abs(base) < 1e-12:
        if abs(target) < 1e-12:
            return 0.0
        return None
    return (target - base) / base * 100.0


def top_skip_reason(skipped_map: Any) -> Optional[str]:
    if not isinstance(skipped_map, dict) or not skipped_map:
        return None
    counts: Dict[str, int] = {}
    for reason in skipped_map.values():
        key = str(reason)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _parse_guard_reason_codes(reason: Optional[str]) -> List[str]:
    if not reason:
        return []
    parsed: List[str] = []
    seen = set()
    for chunk in str(reason).split(";"):
        token = chunk.strip()
        if not token:
            continue
        code = token.split(":", 1)[0].strip()
        if not code:
            continue
        if code not in seen:
            seen.add(code)
            parsed.append(code)
    return parsed


def display_strategy_name(strategy_id: Optional[str]) -> str:
    if not strategy_id:
        return "N/A"
    naming = get_strategy_naming(strategy_id, default_category="unknown")
    return naming["display_name"]


def is_ablation_strategy(strategy_id: str, category: Optional[str] = None) -> bool:
    """判定策略是否属于 KG Component 非核心分支。"""
    naming = get_strategy_naming(strategy_id, default_category=category or "unknown")
    route = naming.get("method_route", "unknown")
    if route == "ablation":
        return True
    # 兼容历史数据：老结果只有 category=kg_component，没有细粒度命名字段。
    return route == "unknown" and (category or "").strip() == "kg_component"


def extract_kg_protocol_mae(kg_task: Dict, canonical_key: str, legacy_key: str) -> Optional[float]:
    """兼容 kg_protocol_* 与 protocol_* 两套键名。"""
    payload = kg_task.get(canonical_key)
    if payload is None:
        payload = kg_task.get(legacy_key, {})
    return extract_mae(payload, "test")


def horizon_key(h: Any) -> str:
    """将 horizon 标准化为字符串键。"""
    if h is None:
        return "unknown"
    if isinstance(h, int):
        return str(h)
    if isinstance(h, float):
        return str(int(h)) if h.is_integer() else str(h)
    s = str(h).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_json_safe(path: Path) -> Optional[Dict]:
    """安全加载 JSON。"""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载失败 {path}: {e}")
        return None


def extract_mae(data: Dict, path: str) -> Optional[float]:
    """提取 MAE。"""
    keys = path.split(".")
    current = data
    for k in keys:
        if current is None or not isinstance(current, dict):
            return None
        current = current.get(k)

    if isinstance(current, dict):
        return current.get("mae")
    return current


def extract_metric(data: Dict, path: str) -> Optional[float]:
    """提取任意指标并转 float。"""
    keys = path.split(".")
    current: Any = data
    for k in keys:
        if current is None or not isinstance(current, dict):
            return None
        current = current.get(k)
    return safe_float(current)


def extract_strategy_slices_from_baseline(task_data: Dict, strategy: str) -> Dict[str, Any]:
    slices_all = (task_data or {}).get("slices", {})
    if not isinstance(slices_all, dict):
        return {}
    payload = slices_all.get(strategy, {})
    return payload if isinstance(payload, dict) else {}


STRATEGY_ALIASES = {
    "simple_avg": ["simple_avg", "simple_avg_safe"],
    "smart_weighted": ["smart_weighted"],
    "static_weight": ["static_weight", "static_weight_safe"],
    "stacking": ["stacking", "stacking_safe"],
    "constrained_opt": ["constrained_opt"],
    "dynamic_avg": ["dynamic_avg"],
    "dynamic_weight": ["dynamic_weight"],
    "dynamic_stacking": ["dynamic_stacking"],
}

_NON_SINGLE_STRATEGIES = {
    "simple_avg", "simple_avg_safe",
    "smart_weighted",
    "static_weight", "static_weight_safe",
    "stacking", "stacking_safe",
    "constrained_opt",
    "dynamic_avg", "dynamic_weight", "dynamic_stacking",
    "rl_qms", "mole_router",
    "dash_tta",
    "gating_network", "soft_gating", "gating_network_v2",
    "scenario_bucket", "adaptive_bucket", "scenario_similarity",
    "kg_protocol_a", "kg_protocol_b", "protocol_A", "protocol_B",
}

_ABLATION_STRATEGIES = {
    "gating_network",
    "soft_gating",
    "scenario_bucket",
    "gating_network_v2",
    "adaptive_bucket",
    "scenario_similarity",
}


def load_base_metrics_by_task(candidates: List[Path]) -> Dict[Tuple[str, str], Dict]:
    """
    从 metrics.json 读取 base 单模型指标（dataset, horizon）-> base 字典。
    """
    for root in candidates:
        if root is None:
            continue
        for metrics_filename in ["metrics.json", "kg_component_metrics.json"]:
            metrics_path = root / metrics_filename
            payload = load_json_safe(metrics_path)
            if not isinstance(payload, dict):
                continue
            by_task: Dict[Tuple[str, str], Dict] = {}
            for ds, ds_data in payload.items():
                if not isinstance(ds_data, dict):
                    continue
                for h, task in ds_data.items():
                    if not isinstance(task, dict):
                        continue
                    base = task.get("base")
                    if isinstance(base, dict):
                        by_task[(str(ds), horizon_key(h))] = base
            if by_task:
                print(f"已加载 base 单模型指标: {metrics_path}")
                return by_task
    return {}


def _best_single_from_base(base: Dict, include_seasonal_naive: bool) -> Tuple[Optional[str], Optional[float]]:
    best_name = None
    best_mae = None
    for model_name, model_payload in (base or {}).items():
        if not include_seasonal_naive and model_name == "seasonal_naive":
            continue
        if not isinstance(model_payload, dict):
            continue
        mae = extract_mae(model_payload, "test")
        if mae is None:
            continue
        try:
            mae = float(mae)
        except Exception:
            continue
        if best_mae is None or mae < best_mae:
            best_name = model_name
            best_mae = mae
    return best_name, best_mae


def _best_single_from_strategy_mae(
    strategy_mae: Dict[str, Any],
    include_seasonal_naive: bool,
) -> Tuple[Optional[str], Optional[float]]:
    best_name = None
    best_mae = None
    for strategy, val in (strategy_mae or {}).items():
        if strategy in _NON_SINGLE_STRATEGIES:
            continue
        if not include_seasonal_naive and strategy == "seasonal_naive":
            continue
        try:
            mae = float(val)
        except Exception:
            continue
        if best_mae is None or mae < best_mae:
            best_name = strategy
            best_mae = mae
    return best_name, best_mae


def _best_eligible_single(
    eligible_models: List[str],
    base_task: Dict,
    strategy_mae: Dict[str, Any],
) -> Tuple[Optional[str], Optional[float]]:
    best_name = None
    best_mae = None
    for model_name in eligible_models or []:
        mae = None
        if isinstance(base_task, dict):
            payload = base_task.get(model_name)
            if isinstance(payload, dict):
                mae = extract_mae(payload, "test")
        if mae is None and isinstance(strategy_mae, dict):
            mae = strategy_mae.get(model_name)
        if mae is None:
            continue
        try:
            mae = float(mae)
        except Exception:
            continue
        if best_mae is None or mae < best_mae:
            best_name = model_name
            best_mae = mae
    return best_name, best_mae


def _best_ablation_from_task_data(task_data: Dict) -> Tuple[Optional[str], Optional[float]]:
    """
    从 task_data["combos"] 中提取最优 ablation 策略（用于无 leaderboard 场景）。
    """
    combos = (task_data or {}).get("combos", {})
    if not isinstance(combos, dict):
        return None, None
    best_name = None
    best_mae = None
    for strategy in _ABLATION_STRATEGIES:
        payload = combos.get(strategy, {})
        if not isinstance(payload, dict):
            continue
        mae = extract_mae(payload, "test")
        if mae is None:
            continue
        try:
            mae = float(mae)
        except Exception:
            continue
        if best_mae is None or mae < best_mae:
            best_name = strategy
            best_mae = mae
    return best_name, best_mae


def extract_strategy_mae_from_baseline(task_data: Dict, strategy: str) -> Optional[float]:
    """从 baseline task 中提取策略 MAE（支持 _safe 别名）。"""
    candidates = STRATEGY_ALIASES.get(strategy, [strategy])
    for strat in candidates:
        val = extract_mae(task_data, f"test.{strat}")
        if val is None and isinstance(task_data, dict):
            # 兼容 kg_component_metrics.json 结构：task_data["combos"][strategy]["test"]["mae"]
            combo_payload = ((task_data.get("combos") or {}).get(strat) or {})
            if isinstance(combo_payload, dict):
                val = extract_mae(combo_payload, "test")
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    return None


def extract_strategy_mae_from_leaderboard(strategy_mae: Dict[str, Any], strategy: str) -> Optional[float]:
    """从 leaderboard 的 strategy_mae 中提取策略 MAE（支持 _safe 别名）。"""
    candidates = STRATEGY_ALIASES.get(strategy, [strategy])
    for strat in candidates:
        if strat in strategy_mae:
            try:
                return float(strategy_mae[strat])
            except Exception:
                continue
    return None


def resolve_leaderboard_path(
    leaderboard_root: Optional[Path],
    baseline_root: Path,
    kg_root: Path,
) -> Optional[Path]:
    """解析 leaderboard_full.csv 路径。"""
    candidates = []
    if leaderboard_root is not None:
        candidates.append(leaderboard_root / "leaderboard_full.csv")
    candidates.append(baseline_root / "leaderboard_full.csv")
    candidates.append(kg_root / "leaderboard_full.csv")
    for p in candidates:
        if p.exists():
            return p
    return None


def load_leaderboard_by_task(path: Path) -> Tuple[Dict[Tuple[str, str], Dict], List[Dict[str, str]]]:
    """读取 leaderboard，并按 (dataset, horizon) 聚合。"""
    issues: List[Dict[str, str]] = []
    by_task: Dict[Tuple[str, str], Dict] = {}
    required_cols = {"dataset", "horizon", "strategy", "category", "test_mae", "status"}

    # 优先使用 pandas；缺失时回退到 csv 标准库
    if pd is not None:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            issues.append({
                "type": "leaderboard_load_error",
                "dataset": "*",
                "horizon": "*",
                "detail": f"{path}: {e}",
            })
            return by_task, issues

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            issues.append({
                "type": "leaderboard_schema_error",
                "dataset": "*",
                "horizon": "*",
                "detail": f"缺少列: {missing}",
            })
            return by_task, issues

        df = df.copy()
        df["horizon"] = df["horizon"].map(horizon_key)
        df["test_mae"] = pd.to_numeric(df["test_mae"], errors="coerce")
        success = df[(df["status"] == "success") & df["test_mae"].notna()]

        for (dataset, h), grp in success.groupby(["dataset", "horizon"]):
            task = {
                "strategy_mae": {},
                "strategy_tail_rmse": {},
                "strategy_test_rmse": {},
                "leaderboard_best_strategy": None,
                "leaderboard_best_mae": None,
                "kg_component_best_strategy": None,
                "kg_component_best_mae": None,
            }

            for _, row in grp.iterrows():
                strat = str(row["strategy"])
                mae = float(row["test_mae"])
                task["strategy_mae"][strat] = mae
                tail_rmse = row.get("tail_rmse")
                if tail_rmse is not None and pd.notna(tail_rmse):
                    try:
                        task["strategy_tail_rmse"][strat] = float(tail_rmse)
                    except Exception:
                        pass
                test_rmse = row.get("test_rmse")
                if test_rmse is not None and pd.notna(test_rmse):
                    try:
                        task["strategy_test_rmse"][strat] = float(test_rmse)
                    except Exception:
                        pass

            idx_all = grp["test_mae"].idxmin()
            row_all = grp.loc[idx_all]
            task["leaderboard_best_strategy"] = str(row_all["strategy"])
            task["leaderboard_best_mae"] = float(row_all["test_mae"])

            kg_component_grp = grp[
                grp.apply(
                    lambda r: is_ablation_strategy(
                        str(r["strategy"]),
                        str(r["category"]),
                    ),
                    axis=1,
                )
            ]
            if not kg_component_grp.empty:
                idx_kg_component = kg_component_grp["test_mae"].idxmin()
                row_kg_component = kg_component_grp.loc[idx_kg_component]
                task["kg_component_best_strategy"] = str(row_kg_component["strategy"])
                task["kg_component_best_mae"] = float(row_kg_component["test_mae"])

            by_task[(str(dataset), str(h))] = task
        return by_task, issues

    # csv fallback
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = set(reader.fieldnames or [])
            missing = [c for c in required_cols if c not in header]
            if missing:
                issues.append({
                    "type": "leaderboard_schema_error",
                    "dataset": "*",
                    "horizon": "*",
                    "detail": f"缺少列: {missing}",
                })
                return by_task, issues

            grouped_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for row in reader:
                if row.get("status") != "success":
                    continue
                try:
                    mae = float(row.get("test_mae", "nan"))
                except Exception:
                    continue
                if mae != mae:  # NaN
                    continue
                ds = str(row.get("dataset", "")).strip()
                h = horizon_key(row.get("horizon"))
                key = (ds, h)
                grouped_rows.setdefault(key, []).append({
                    "strategy": str(row.get("strategy", "")).strip(),
                    "category": str(row.get("category", "")).strip(),
                    "test_mae": mae,
                    "tail_rmse": row.get("tail_rmse"),
                    "test_rmse": row.get("test_rmse"),
                })

        for key, rows in grouped_rows.items():
            task = {
                "strategy_mae": {},
                "strategy_tail_rmse": {},
                "strategy_test_rmse": {},
                "leaderboard_best_strategy": None,
                "leaderboard_best_mae": None,
                "kg_component_best_strategy": None,
                "kg_component_best_mae": None,
            }
            for r in rows:
                task["strategy_mae"][r["strategy"]] = float(r["test_mae"])
                try:
                    tail_rmse = r.get("tail_rmse")
                    if tail_rmse is not None and str(tail_rmse).strip() != "":
                        task["strategy_tail_rmse"][r["strategy"]] = float(tail_rmse)
                except Exception:
                    pass
                try:
                    test_rmse = r.get("test_rmse")
                    if test_rmse is not None and str(test_rmse).strip() != "":
                        task["strategy_test_rmse"][r["strategy"]] = float(test_rmse)
                except Exception:
                    pass
            best_all = min(rows, key=lambda x: x["test_mae"])
            task["leaderboard_best_strategy"] = best_all["strategy"]
            task["leaderboard_best_mae"] = float(best_all["test_mae"])

            kg_component_rows = [
                r for r in rows
                if is_ablation_strategy(r["strategy"], r["category"])
            ]
            if kg_component_rows:
                best_kg_component = min(kg_component_rows, key=lambda x: x["test_mae"])
                task["kg_component_best_strategy"] = best_kg_component["strategy"]
                task["kg_component_best_mae"] = float(best_kg_component["test_mae"])

            by_task[key] = task
    except Exception as e:
        issues.append({
            "type": "leaderboard_load_error",
            "dataset": "*",
            "horizon": "*",
            "detail": f"{path}: {e}",
        })

    return by_task, issues


def collect_task_pairs(
    baseline_results: Dict,
    kg_results: Dict,
    leaderboard_task_map: Dict[Tuple[str, str], Dict],
) -> List[Tuple[str, str]]:
    """收集所有任务键。"""
    pairs = set()
    for ds, hs in DATASET_HORIZONS.items():
        for h in hs:
            pairs.add((ds, horizon_key(h)))

    for ds, ds_data in (baseline_results or {}).items():
        if isinstance(ds_data, dict):
            for h in ds_data.keys():
                pairs.add((str(ds), horizon_key(h)))

    for ds, ds_data in (kg_results or {}).items():
        if isinstance(ds_data, dict):
            for h in ds_data.keys():
                pairs.add((str(ds), horizon_key(h)))

    for ds, h in leaderboard_task_map.keys():
        pairs.add((str(ds), horizon_key(h)))

    ds_order = {ds: i for i, ds in enumerate(DATASET_HORIZONS.keys())}

    def _sort_key(pair: Tuple[str, str]):
        ds, h = pair
        try:
            h_num = int(float(h))
        except Exception:
            h_num = 10**9
        return (ds_order.get(ds, 10**6), ds, h_num)

    return sorted(pairs, key=_sort_key)


def compute_consistency_issues(
    task_pairs: List[Tuple[str, str]],
    baseline_results: Dict,
    kg_results: Dict,
    leaderboard_task_map: Dict[Tuple[str, str], Dict],
) -> List[Dict[str, str]]:
    """计算跨来源口径一致性问题。"""
    issues: List[Dict[str, str]] = []

    baseline_pairs = {
        (str(ds), horizon_key(h))
        for ds, ds_data in (baseline_results or {}).items()
        for h in (ds_data.keys() if isinstance(ds_data, dict) else [])
    }
    kg_pairs = {
        (str(ds), horizon_key(h))
        for ds, ds_data in (kg_results or {}).items()
        for h in (ds_data.keys() if isinstance(ds_data, dict) else [])
    }
    leaderboard_pairs = set(leaderboard_task_map.keys())

    for ds, h in task_pairs:
        present = []
        if (ds, h) in baseline_pairs:
            present.append("baseline")
        if (ds, h) in kg_pairs:
            present.append("kg")
        if (ds, h) in leaderboard_pairs:
            present.append("leaderboard")
        if len(present) < 2:
            issues.append({
                "type": "source_coverage_gap",
                "dataset": ds,
                "horizon": h,
                "detail": f"仅来源 {present if present else ['none']}",
            })

    # 同一任务上 baseline 与 leaderboard 的同名策略数值一致性检查
    cmp_strategies = [
        "simple_avg", "smart_weighted", "static_weight", "stacking", "constrained_opt",
        "dynamic_avg", "dynamic_weight", "dynamic_stacking",
    ]
    for ds, h in task_pairs:
        if (ds, h) not in baseline_pairs or (ds, h) not in leaderboard_pairs:
            continue
        bl = baseline_results.get(ds, {}).get(h, {})
        lb = leaderboard_task_map.get((ds, h), {})
        lb_mae = lb.get("strategy_mae", {})
        for strat in cmp_strategies:
            bl_mae = extract_strategy_mae_from_baseline(bl, strat)
            lb_v = extract_strategy_mae_from_leaderboard(lb_mae, strat)
            if bl_mae is None or lb_v is None:
                continue
            if abs(float(bl_mae) - float(lb_v)) > 1e-6:
                issues.append({
                    "type": "metric_mismatch",
                    "dataset": ds,
                    "horizon": h,
                    "detail": f"{strat}: baseline={bl_mae:.6f}, leaderboard={lb_v:.6f}",
                })

    return issues


def generate_comparison_report(
    baseline_root: Path,
    kg_root: Path,
    out_path: Path,
    leaderboard_root: Optional[Path] = None,
    datasets: Optional[List[str]] = None,
):
    """生成统一对比报告。"""
    baseline_candidates = [baseline_root]
    if leaderboard_root is not None and leaderboard_root not in baseline_candidates:
        baseline_candidates.append(leaderboard_root)

    baseline_results = None
    for cand in baseline_candidates:
        for bl_filename in ["combo_results.json", "kg_component_metrics.json", "metrics.json"]:
            baseline_results = load_json_safe(cand / bl_filename)
            if baseline_results is not None:
                print(f"已加载 baseline 组合结果: {cand / bl_filename}")
                break
        if baseline_results is not None:
            break

    if baseline_results is None:
        baseline_results = {}
        for cand in baseline_candidates:
            val_metrics = load_json_safe(cand / "val_combo_metrics.json")
            test_metrics = load_json_safe(cand / "test_combo_metrics.json")
            if not (val_metrics and test_metrics):
                continue
            print(f"已加载 baseline 指标: {cand / 'test_combo_metrics.json'}")
            for ds in set(list(val_metrics.keys()) + list(test_metrics.keys())):
                baseline_results.setdefault(ds, {})
                for h in set(list(val_metrics.get(ds, {}).keys()) + list(test_metrics.get(ds, {}).keys())):
                    baseline_results[ds][str(h)] = {
                        "val_insample": val_metrics.get(ds, {}).get(h, {}),
                        "test": test_metrics.get(ds, {}).get(h, {}),
                    }
            break

    kg_results = None
    for kg_filename in ["kg_results.json", "kg_comparison.json", "kg_metrics.json", "combo_results.json"]:
        kg_results = load_json_safe(kg_root / kg_filename)
        if kg_results is not None:
            print(f"已加载 KG 结果: {kg_root / kg_filename}")
            break
    if kg_results is None:
        print(f"警告: 未找到 KG 结果文件于 {kg_root}")
        kg_results = {}

    leaderboard_path = resolve_leaderboard_path(leaderboard_root, baseline_root, kg_root)
    leaderboard_task_map: Dict[Tuple[str, str], Dict] = {}
    parse_issues: List[Dict[str, str]] = []
    if leaderboard_path is not None:
        print(f"已加载 leaderboard: {leaderboard_path}")
        leaderboard_task_map, parse_issues = load_leaderboard_by_task(leaderboard_path)
    else:
        print("警告: 未找到 leaderboard_full.csv，KG Component 对比将显示为 N/A")

    metrics_candidates = []
    if leaderboard_root is not None:
        metrics_candidates.append(leaderboard_root)
    if baseline_root not in metrics_candidates:
        metrics_candidates.append(baseline_root)
    if kg_root not in metrics_candidates:
        metrics_candidates.append(kg_root)
    base_metrics_by_task = load_base_metrics_by_task(metrics_candidates)

    if datasets is None:
        datasets = ["pjm", "aemo_vic", "aemo_nsw"]
    allowed_datasets = {str(ds) for ds in datasets}

    task_pairs = collect_task_pairs(baseline_results, kg_results, leaderboard_task_map)
    task_pairs = [pair for pair in task_pairs if pair[0] in allowed_datasets]
    consistency_issues = parse_issues + compute_consistency_issues(
        task_pairs, baseline_results, kg_results, leaderboard_task_map
    )

    summary = []
    for dataset, h_str in task_pairs:
        row = {
            "dataset": dataset,
            "horizon": int(float(h_str)) if h_str.replace(".", "", 1).isdigit() else h_str,
        }

        lb_task = leaderboard_task_map.get((dataset, h_str), {})
        lb_strategy_mae = lb_task.get("strategy_mae", {})
        bl = baseline_results.get(dataset, {}).get(h_str, {})
        for strategy in [
            "simple_avg", "smart_weighted", "static_weight", "stacking", "constrained_opt",
            "dynamic_avg", "dynamic_weight", "dynamic_stacking",
        ]:
            val = extract_strategy_mae_from_baseline(bl, strategy)
            if val is None:
                val = extract_strategy_mae_from_leaderboard(lb_strategy_mae, strategy)
            row[f"baseline_{strategy}"] = val
        for field, strat in [
            ("competitor_rl_qms", "rl_qms"),
            ("competitor_mole_router", "mole_router"),
            ("ablation_scenario_bucket", "scenario_bucket"),
            ("ablation_scenario_similarity", "scenario_similarity"),
        ]:
            val = extract_strategy_mae_from_baseline(bl, strat)
            if val is None:
                val = extract_strategy_mae_from_leaderboard(lb_strategy_mae, strat)
            row[field] = val

        base_task = base_metrics_by_task.get((dataset, h_str), {})
        itr_mae = None
        if isinstance(base_task, dict):
            itr_payload = base_task.get("itransformer", {})
            if isinstance(itr_payload, dict):
                itr_mae = extract_mae(itr_payload, "test")
        if itr_mae is None:
            itr_mae = extract_strategy_mae_from_baseline(bl, "itransformer")
        if itr_mae is None:
            itr_mae = extract_strategy_mae_from_leaderboard(lb_strategy_mae, "itransformer")
        row["competitor_itransformer"] = safe_float(itr_mae)

        kg = kg_results.get(dataset, {}).get(h_str, {})
        row["kg_protocol_a"] = extract_kg_protocol_mae(kg, "kg_protocol_a", "protocol_A")
        row["kg_protocol_b"] = extract_kg_protocol_mae(kg, "kg_protocol_b", "protocol_B")
        kg_a_payload = kg.get("kg_protocol_a")
        if kg_a_payload is None:
            kg_a_payload = kg.get("protocol_A", {})
        kg_b_payload = kg.get("kg_protocol_b")
        if kg_b_payload is None:
            kg_b_payload = kg.get("protocol_B", {})
        if not isinstance(kg_a_payload, dict):
            kg_a_payload = {}
        if not isinstance(kg_b_payload, dict):
            kg_b_payload = {}
        row["kg_protocol_a_protocol"] = kg_a_payload.get("protocol")
        row["kg_protocol_b_protocol"] = kg_b_payload.get("protocol")
        for metric_name in CORE_METRICS + ROBUST_METRICS:
            row[f"kg_protocol_a_{metric_name}"] = extract_metric(kg_a_payload, f"test.{metric_name}")
            row[f"kg_protocol_b_{metric_name}"] = extract_metric(kg_b_payload, f"test.{metric_name}")
        selected_a = (kg_a_payload.get("test") or {}).get("selected_models") if isinstance(kg_a_payload.get("test"), dict) else None
        selected_b = (kg_b_payload.get("test") or {}).get("selected_models") if isinstance(kg_b_payload.get("test"), dict) else None
        row["kg_protocol_a_selected_models"] = selected_a if isinstance(selected_a, list) else []
        row["kg_protocol_b_selected_models"] = selected_b if isinstance(selected_b, list) else []
        row["kg_protocol_a_selected_count"] = len(row["kg_protocol_a_selected_models"])
        row["kg_protocol_b_selected_count"] = len(row["kg_protocol_b_selected_models"])
        row["kg_protocol_b_error"] = kg_b_payload.get("error")
        row["kg_protocol_b_runtime_ok"] = row["kg_protocol_b_error"] is None
        kg_meta = kg.get("_meta", {}) if isinstance(kg, dict) else {}
        kg_b_val = kg_b_payload.get("val", {}) if isinstance(kg_b_payload, dict) else {}
        kg_b_weight_meta = kg_b_val.get("weight_meta", {}) if isinstance(kg_b_val, dict) else {}
        if not isinstance(kg_b_weight_meta, dict):
            kg_b_weight_meta = {}
        guard_meta = kg_b_weight_meta.get("protocol_b_guard", {})
        guard_cfg = kg_b_weight_meta.get("guard_config", {})
        if not isinstance(guard_meta, dict):
            guard_meta = {}
        if not isinstance(guard_cfg, dict):
            guard_cfg = {}
        guard_reason = guard_meta.get("reason") or guard_cfg.get("final_fallback_reason")
        guard_target = guard_meta.get("fallback_target") or guard_cfg.get("final_fallback_target")
        guard_codes = _parse_guard_reason_codes(guard_reason)
        row["protocol_b_guard_reason"] = guard_reason
        row["protocol_b_guard_fallback_target"] = guard_target
        row["protocol_b_guard_reason_codes"] = guard_codes
        row["protocol_b_guard_triggered"] = bool(guard_reason)
        pb_proto = str(row.get("kg_protocol_b_protocol") or "")
        row["protocol_b_engaged"] = bool(
            row.get("kg_protocol_b_runtime_ok")
            and pb_proto
            and "fallback" not in pb_proto.lower()
        )
        row["runtime_protocol_a_sec"] = (
            kg_meta.get("runtime_protocol_a_sec") if isinstance(kg_meta, dict) else None
        )
        row["runtime_protocol_b_sec"] = (
            kg_meta.get("runtime_protocol_b_sec") if isinstance(kg_meta, dict) else None
        )
        safe_models = kg_meta.get("safe_models", []) if isinstance(kg_meta, dict) else []
        row["safe_models_count"] = len(safe_models) if isinstance(safe_models, list) else None
        row["candidate_models_count"] = (
            kg_meta.get("candidate_models_count") if isinstance(kg_meta, dict) else None
        )
        row["common_models_count"] = (
            kg_meta.get("common_models_count") if isinstance(kg_meta, dict) else None
        )
        if row["common_models_count"] not in (None, 0):
            row["kg_protocol_b_model_ratio"] = (
                float(row["kg_protocol_b_selected_count"]) / float(row["common_models_count"])
            )
        else:
            row["kg_protocol_b_model_ratio"] = None
        row["pool_mode"] = kg_meta.get("pool_mode") if isinstance(kg_meta, dict) else None
        row["naive_selected_by_protocol_b"] = (
            bool(kg_meta.get("naive_selected_by_protocol_b")) if isinstance(kg_meta, dict) else False
        )
        row["naive_selected_ratio_protocol_b"] = (
            kg_meta.get("naive_selected_ratio_protocol_b") if isinstance(kg_meta, dict) else None
        )
        interaction_branch = kg_b_weight_meta.get("interaction_branch", {})
        if not isinstance(interaction_branch, dict):
            interaction_branch = {}
        row["interaction_disabled_reason"] = (
            interaction_branch.get("disabled_reason_code")
            or kg_b_weight_meta.get("interaction_disable_reason")
        )
        row["interaction_reject_reasons"] = (
            interaction_branch.get("reject_reasons")
            if isinstance(interaction_branch.get("reject_reasons"), list)
            else []
        )
        row["interaction_applied"] = bool(interaction_branch.get("applied"))
        ext_meta = kg_meta.get("extended_pool", {}) if isinstance(kg_meta, dict) else {}
        ext_val = ext_meta.get("val", {}) if isinstance(ext_meta, dict) else {}
        ext_test = ext_meta.get("test", {}) if isinstance(ext_meta, dict) else {}
        row["ext_pool_val_loaded_count"] = (
            ext_meta.get("val_loaded_count")
            if isinstance(ext_meta.get("val_loaded_count"), int)
            else len(ext_val.get("loaded", [])) if isinstance(ext_val, dict) else None
        )
        row["ext_pool_test_loaded_count"] = (
            ext_meta.get("test_loaded_count")
            if isinstance(ext_meta.get("test_loaded_count"), int)
            else len(ext_test.get("loaded", [])) if isinstance(ext_test, dict) else None
        )
        row["ext_pool_common_loaded_count"] = (
            ext_meta.get("common_loaded_count")
            if isinstance(ext_meta.get("common_loaded_count"), int)
            else None
        )
        row["ext_pool_top_skip_reason"] = top_skip_reason(
            ext_val.get("skipped", {}) if isinstance(ext_val, dict) else {}
        )
        audit_meta = kg_meta.get("candidate_audit", {}) if isinstance(kg_meta, dict) else {}
        accepted_models = audit_meta.get("accepted_models", []) if isinstance(audit_meta, dict) else []
        removed_models = audit_meta.get("removed_models", []) if isinstance(audit_meta, dict) else []
        soft_passed_models = audit_meta.get("soft_passed_models", []) if isinstance(audit_meta, dict) else []
        removed_unaudited_models = (
            audit_meta.get("removed_unaudited_models", []) if isinstance(audit_meta, dict) else []
        )
        row["candidate_audit_applied"] = bool(audit_meta.get("applied")) if isinstance(audit_meta, dict) else False
        row["candidate_audit_accepted_count"] = len(accepted_models) if isinstance(accepted_models, list) else None
        row["candidate_audit_removed_count"] = len(removed_models) if isinstance(removed_models, list) else None
        row["candidate_audit_soft_pass_enabled"] = (
            bool(audit_meta.get("soft_pass_enabled")) if isinstance(audit_meta, dict) else None
        )
        row["candidate_audit_soft_pass_count"] = (
            len(soft_passed_models) if isinstance(soft_passed_models, list) else None
        )
        row["candidate_audit_removed_unaudited_count"] = (
            len(removed_unaudited_models) if isinstance(removed_unaudited_models, list) else None
        )
        row["candidate_audit_top_soft_pass"] = (
            soft_passed_models[0] if isinstance(soft_passed_models, list) and soft_passed_models else None
        )
        row["candidate_audit_top_removed"] = (
            removed_models[0] if isinstance(removed_models, list) and removed_models else None
        )
        row["candidate_audit_skip_reason"] = (
            audit_meta.get("skip_reason") if isinstance(audit_meta, dict) else None
        )

        best_single_excl_name, best_single_excl_mae = _best_single_from_base(
            base_task, include_seasonal_naive=False
        )
        best_single_incl_name, best_single_incl_mae = _best_single_from_base(
            base_task, include_seasonal_naive=True
        )
        if best_single_excl_name is None:
            best_single_excl_name, best_single_excl_mae = _best_single_from_strategy_mae(
                lb_task.get("strategy_mae", {}),
                include_seasonal_naive=False,
            )
        if best_single_incl_name is None:
            best_single_incl_name, best_single_incl_mae = _best_single_from_strategy_mae(
                lb_task.get("strategy_mae", {}),
                include_seasonal_naive=True,
            )
        row["best_single_excl_naive_strategy"] = best_single_excl_name
        row["best_single_excl_naive_mae"] = best_single_excl_mae
        row["best_single_incl_naive_strategy"] = best_single_incl_name
        row["best_single_incl_naive_mae"] = best_single_incl_mae
        eligible_models = []
        if isinstance(kg_meta, dict):
            eligible_models = kg_meta.get("eligible_models", []) or []
        eligible_name, eligible_mae = _best_eligible_single(
            eligible_models=eligible_models,
            base_task=base_task,
            strategy_mae=lb_task.get("strategy_mae", {}),
        )
        row["best_eligible_single_strategy"] = eligible_name
        row["best_eligible_single_mae"] = eligible_mae
        row["eligible_models_count"] = len(eligible_models) if isinstance(eligible_models, list) else None

        row["kg_component_best_strategy"] = lb_task.get("kg_component_best_strategy")
        row["kg_component_best_mae"] = lb_task.get("kg_component_best_mae")
        if row["kg_component_best_strategy"] is None:
            ab_name, ab_mae = _best_ablation_from_task_data(bl)
            row["kg_component_best_strategy"] = ab_name
            row["kg_component_best_mae"] = ab_mae
        row["leaderboard_best_strategy"] = lb_task.get("leaderboard_best_strategy")
        row["leaderboard_best_mae"] = lb_task.get("leaderboard_best_mae")

        # 三类 gap 诊断：
        # 1) val->test: best single
        # 2) cv->test: kg component best（这里 cv 口径使用该策略 val 指标）
        # 3) test<-tail RMSE: kg component best（leaderboard 的 tail_rmse / test_rmse）
        row["gap_val_test_best_single"] = None
        row["gap_cv_test_kg_component_best"] = None
        row["gap_test_vs_tail_rmse_kg_component"] = None
        # 兼容旧字段名
        row["gap_tail_test_kg_component"] = None

        if row.get("best_single_excl_naive_strategy") and isinstance(base_task, dict):
            best_single_payload = base_task.get(row["best_single_excl_naive_strategy"], {})
            if isinstance(best_single_payload, dict):
                bs_val_mae = extract_mae(best_single_payload, "val")
                bs_test_mae = extract_mae(best_single_payload, "test")
                row["gap_val_test_best_single"] = gap_pct(bs_val_mae, bs_test_mae)
                for metric_name in CORE_METRICS + ROBUST_METRICS:
                    row[f"best_single_excl_naive_{metric_name}"] = extract_metric(
                        best_single_payload, f"test.{metric_name}"
                    )
            bs_slices = extract_strategy_slices_from_baseline(
                bl,
                row.get("best_single_excl_naive_strategy"),
            )
            for slice_key in SLICE_KEYS:
                slice_payload = bs_slices.get(slice_key, {}) if isinstance(bs_slices, dict) else {}
                if isinstance(slice_payload, dict):
                    row[f"best_single_excl_{slice_key}_mae"] = safe_float(slice_payload.get("mae"))
                    row[f"best_single_excl_{slice_key}_rmse"] = safe_float(slice_payload.get("rmse"))

        if row.get("best_single_incl_naive_strategy") and isinstance(base_task, dict):
            best_single_payload = base_task.get(row["best_single_incl_naive_strategy"], {})
            if isinstance(best_single_payload, dict):
                for metric_name in CORE_METRICS + ROBUST_METRICS:
                    row[f"best_single_incl_naive_{metric_name}"] = extract_metric(
                        best_single_payload, f"test.{metric_name}"
                    )

        # KG-B 可直接导出的切片（head/mid/tail）
        if isinstance(((kg_b_payload.get("test") or {}).get("slice_evidence") or {}).get("temporal_slices"), dict):
            temporal_payload = ((kg_b_payload.get("test") or {}).get("slice_evidence") or {}).get("temporal_slices")
            for slice_key in ["head", "mid", "tail"]:
                s = temporal_payload.get(slice_key, {})
                if isinstance(s, dict):
                    row[f"kg_protocol_b_temporal_{slice_key}_mae"] = safe_float(s.get("mae"))
                    row[f"kg_protocol_b_temporal_{slice_key}_n"] = int(s.get("n")) if s.get("n") is not None else None

        kg_component_best_strategy = row.get("kg_component_best_strategy")
        combos_data = bl.get("combos", {}) if isinstance(bl, dict) else {}
        if kg_component_best_strategy and isinstance(combos_data, dict):
            kg_component_payload = combos_data.get(kg_component_best_strategy, {})
            if isinstance(kg_component_payload, dict):
                kg_component_val_mae = extract_mae(kg_component_payload, "val")
                kg_component_test_mae = extract_mae(kg_component_payload, "test")
                row["gap_cv_test_kg_component_best"] = gap_pct(kg_component_val_mae, kg_component_test_mae)
        if kg_component_best_strategy:
            kg_component_slices = extract_strategy_slices_from_baseline(bl, kg_component_best_strategy)
            for slice_key in SLICE_KEYS:
                sp = kg_component_slices.get(slice_key, {}) if isinstance(kg_component_slices, dict) else {}
                if isinstance(sp, dict):
                    row[f"kg_component_best_{slice_key}_mae"] = safe_float(sp.get("mae"))
                    row[f"kg_component_best_{slice_key}_rmse"] = safe_float(sp.get("rmse"))

        if kg_component_best_strategy:
            tail_map = lb_task.get("strategy_tail_rmse", {}) if isinstance(lb_task, dict) else {}
            test_map = lb_task.get("strategy_test_rmse", {}) if isinstance(lb_task, dict) else {}
            tail_rmse = tail_map.get(kg_component_best_strategy)
            test_rmse = test_map.get(kg_component_best_strategy)
            row["gap_test_vs_tail_rmse_kg_component"] = gap_pct(tail_rmse, test_rmse)
            row["gap_tail_test_kg_component"] = row["gap_test_vs_tail_rmse_kg_component"]

        candidates: List[Tuple[str, float]] = []
        for key in [
            "baseline_simple_avg", "baseline_smart_weighted", "baseline_static_weight",
            "baseline_stacking", "baseline_constrained_opt",
            "baseline_dynamic_avg", "baseline_dynamic_weight", "baseline_dynamic_stacking",
            "competitor_rl_qms", "competitor_mole_router", "competitor_itransformer",
            "ablation_scenario_bucket", "ablation_scenario_similarity",
            "kg_protocol_a", "kg_protocol_b",
        ]:
            val = row.get(key)
            if val is not None:
                candidates.append((key, float(val)))
        if row.get("best_single_excl_naive_mae") is not None:
            candidates.append(("best_single_excl_naive", float(row["best_single_excl_naive_mae"])))
        if row.get("best_single_incl_naive_mae") is not None:
            candidates.append(("best_single_incl_naive", float(row["best_single_incl_naive_mae"])))
        if row.get("kg_component_best_strategy") and row.get("kg_component_best_mae") is not None:
            candidates.append((
                f"kg_component_{row['kg_component_best_strategy']}",
                float(row["kg_component_best_mae"]),
            ))

        if candidates:
            best_name, best_mae = min(candidates, key=lambda x: x[1])
            row["best_strategy"] = best_name
            row["best_mae"] = best_mae
        else:
            row["best_strategy"] = None
            row["best_mae"] = None

        summary.append(row)

    expected_tasks = 0
    for ds in allowed_datasets:
        expected_tasks += len(DATASET_HORIZONS.get(ds, []))
    valid_tasks = sum(1 for row in summary if row.get("best_mae") is not None)
    kg_valid_tasks = sum(
        1
        for row in summary
        if row.get("kg_protocol_a") is not None or row.get("kg_protocol_b") is not None
    )
    kg_protocol_b_runtime_ok_tasks = sum(
        1 for row in summary if row.get("kg_protocol_b_runtime_ok") is True
    )
    guard_reason_counts: Dict[str, int] = {}
    guard_reason_tasks: Dict[str, List[str]] = {}
    interaction_disabled_counts: Dict[str, int] = {}
    interaction_reject_counts: Dict[str, int] = {}
    pool_mode_counts: Dict[str, int] = {}
    for row in summary:
        codes = row.get("protocol_b_guard_reason_codes") or []
        if not codes:
            task_id = None
        else:
            task_id = f"{row['dataset']} h={row['horizon']}"
            for code in codes:
                code_key = str(code)
                guard_reason_counts[code_key] = guard_reason_counts.get(code_key, 0) + 1
                guard_reason_tasks.setdefault(code_key, []).append(task_id)
        pool_mode = row.get("pool_mode")
        if pool_mode:
            pool_mode_key = str(pool_mode)
            pool_mode_counts[pool_mode_key] = pool_mode_counts.get(pool_mode_key, 0) + 1
        inter_disabled = row.get("interaction_disabled_reason")
        if inter_disabled:
            key = str(inter_disabled)
            interaction_disabled_counts[key] = interaction_disabled_counts.get(key, 0) + 1
        for reason in row.get("interaction_reject_reasons") or []:
            reason_key = str(reason)
            interaction_reject_counts[reason_key] = interaction_reject_counts.get(reason_key, 0) + 1

    report_lines = [
        "# 统一组合策略对比报告",
        "",
        f"- baseline_root: `{baseline_root}`",
        f"- kg_root: `{kg_root}`",
        f"- leaderboard: `{leaderboard_path}`" if leaderboard_path else "- leaderboard: `N/A`",
        f"- datasets: `{sorted(allowed_datasets)}`",
        f"- expected_tasks: `{expected_tasks}`",
        f"- valid_tasks: `{valid_tasks}`",
        f"- kg_valid_tasks: `{kg_valid_tasks}`",
        f"- kg_protocol_b_runtime_ok_tasks: `{kg_protocol_b_runtime_ok_tasks}`",
        "",
        "## 主报告（KG + SOTA + Perception Ablation）",
        "",
        "| Dataset | H | Best Eligible Single | Best Single (No Naive) | Best Single (With Naive) | RL-QMS | MoLE | iTransformer | KG-A | KG-B | KG Best | scenario_bucket | scenario_similarity | Gap val->test (Best Single) | Gap cv->test (KG Component Best) | Gap test<-tail RMSE (KG Component Best) | ExtPool loaded(val/test/common) | ExtPool top skip reason |",
        "|---------|---|----------------------|------------------------|---------------------------|--------|------|--------------|------|------|---------|-----------------|---------------------|-----------------------------|---------------------------|----------------------------------|-------------------------------|-------------------------|",
    ]

    for row in summary:
        bs_excl = "N/A"
        if row.get("best_single_excl_naive_strategy") and row.get("best_single_excl_naive_mae") is not None:
            bs_excl = (
                f"{row['best_single_excl_naive_strategy']} "
                f"({safe_format(row['best_single_excl_naive_mae'])})"
            )
        bs_incl = "N/A"
        if row.get("best_single_incl_naive_strategy") and row.get("best_single_incl_naive_mae") is not None:
            bs_incl = (
                f"{row['best_single_incl_naive_strategy']} "
                f"({safe_format(row['best_single_incl_naive_mae'])})"
            )
        bs_eligible = "N/A"
        if row.get("best_eligible_single_strategy") and row.get("best_eligible_single_mae") is not None:
            bs_eligible = (
                f"{row['best_eligible_single_strategy']} "
                f"({safe_format(row['best_eligible_single_mae'])})"
            )
        kg_best_name = None
        kg_best_val = None
        for kg_key, kg_label in [("kg_protocol_a", "KG-A"), ("kg_protocol_b", "KG-B")]:
            val = row.get(kg_key)
            if val is not None and (kg_best_val is None or val < kg_best_val):
                kg_best_val = float(val)
                kg_best_name = kg_label
        kg_best_display = f"{kg_best_name} ({safe_format(kg_best_val)})" if kg_best_name else "N/A"
        report_lines.append(
            f"| {row['dataset']} | {row['horizon']} | {bs_eligible} | {bs_excl} | {bs_incl} | "
            f"{safe_format(row.get('competitor_rl_qms'))} | {safe_format(row.get('competitor_mole_router'))} | "
            f"{safe_format(row.get('competitor_itransformer'))} | "
            f"{safe_format(row.get('kg_protocol_a'))} | {safe_format(row.get('kg_protocol_b'))} | "
            f"{kg_best_display} | {safe_format(row.get('ablation_scenario_bucket'))} | "
            f"{safe_format(row.get('ablation_scenario_similarity'))} | "
            f"{safe_pct(row.get('gap_val_test_best_single'))} | "
            f"{safe_pct(row.get('gap_cv_test_kg_component_best'))} | "
            f"{safe_pct(row.get('gap_test_vs_tail_rmse_kg_component'))} | "
            f"{row.get('ext_pool_val_loaded_count', 'N/A')}/"
            f"{row.get('ext_pool_test_loaded_count', 'N/A')}/"
            f"{row.get('ext_pool_common_loaded_count', 'N/A')} | "
            f"{row.get('ext_pool_top_skip_reason') or 'N/A'} |"
        )

    report_lines.extend([
        "",
        "## 附录：全策略 MAE 汇总",
        "",
        "| Dataset | H | Simple Avg | Smart | Static | Stacking | Constrained | KG-A | KG-B | KG Component Best | Best |",
        "|---------|---|------------|-------|--------|----------|-------------|------|------|-----------|------|",
    ])

    for row in summary:
        kg_component_best_display = "N/A"
        if row.get("kg_component_best_strategy") and row.get("kg_component_best_mae") is not None:
            kg_component_best_display = (
                f"{display_strategy_name(row['kg_component_best_strategy'])} "
                f"({safe_format(row['kg_component_best_mae'])})"
            )
        best = row.get("best_strategy") or "N/A"
        best = best.replace("baseline_", "").replace("kg_", "").replace("kg_component_", "")
        best = display_strategy_name(best)
        line = (
            f"| {row['dataset']} | {row['horizon']} | "
            f"{safe_format(row.get('baseline_simple_avg'))} | "
            f"{safe_format(row.get('baseline_smart_weighted'))} | "
            f"{safe_format(row.get('baseline_static_weight'))} | "
            f"{safe_format(row.get('baseline_stacking'))} | "
            f"{safe_format(row.get('baseline_constrained_opt'))} | "
            f"{safe_format(row.get('kg_protocol_a'))} | "
            f"{safe_format(row.get('kg_protocol_b'))} | "
            f"{kg_component_best_display} | {best} |"
        )
        report_lines.append(line)

    report_lines.extend(["", "## 策略获胜次数", ""])
    win_counts: Dict[str, int] = {}
    for row in summary:
        best = row.get("best_strategy")
        if best:
            win_counts[best] = win_counts.get(best, 0) + 1
    for strategy, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        display_id = strategy.replace("baseline_", "").replace("kg_", "").replace("kg_component_", "")
        report_lines.append(f"- **{display_strategy_name(display_id)}** (`{strategy}`): {count} 次")

    improvements = []
    kg_component_gaps = []
    for row in summary:
        baseline_best = None
        for s in [
            "baseline_static_weight", "baseline_stacking", "baseline_constrained_opt",
            "baseline_dynamic_avg", "baseline_dynamic_weight", "baseline_dynamic_stacking",
        ]:
            val = row.get(s)
            if val is not None and (baseline_best is None or val < baseline_best):
                baseline_best = val

        kg_best = None
        for s in ["kg_protocol_a", "kg_protocol_b"]:
            val = row.get(s)
            if val is not None and (kg_best is None or val < kg_best):
                kg_best = val

        if baseline_best is not None and kg_best is not None:
            improvement = (baseline_best - kg_best) / (baseline_best + 1e-10) * 100
            improvements.append({
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                "baseline_best": baseline_best,
                "kg_best": kg_best,
                "improvement_pct": improvement,
            })

        kg_component_best = row.get("kg_component_best_mae")
        if baseline_best is not None and kg_component_best is not None:
            kg_component_gap = (kg_component_best - baseline_best) / (baseline_best + 1e-10) * 100
            kg_component_gaps.append({
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                "baseline_best": baseline_best,
                "kg_component_best_strategy": row.get("kg_component_best_strategy"),
                "kg_component_best": kg_component_best,
                "gap_pct": kg_component_gap,
            })

    catastrophic_lt_3 = sum(1 for i in improvements if i["improvement_pct"] < -3.0)
    catastrophic_lt_5 = sum(1 for i in improvements if i["improvement_pct"] < -5.0)
    safe_counts = [float(r["safe_models_count"]) for r in summary if r.get("safe_models_count") is not None]
    common_counts = [float(r["common_models_count"]) for r in summary if r.get("common_models_count") is not None]
    candidate_counts = [float(r["candidate_models_count"]) for r in summary if r.get("candidate_models_count") is not None]
    runtime_pairs = [
        (float(r["runtime_protocol_a_sec"]), float(r["runtime_protocol_b_sec"]))
        for r in summary
        if r.get("runtime_protocol_a_sec") is not None and r.get("runtime_protocol_b_sec") is not None
    ]
    protocol_b_engaged_count = sum(1 for r in summary if r.get("protocol_b_engaged") is True)
    protocol_b_fallback_count = sum(1 for r in summary if r.get("protocol_b_guard_triggered") is True)
    fallback_target_counts: Dict[str, int] = {}
    for r in summary:
        if not r.get("protocol_b_guard_triggered"):
            continue
        t = str(r.get("protocol_b_guard_fallback_target") or "unknown")
        fallback_target_counts[t] = fallback_target_counts.get(t, 0) + 1

    core_metric_avg = {}
    for m in CORE_METRICS:
        core_metric_avg[f"kg_protocol_b_{m}"] = mean_or_none(
            [safe_float(r.get(f"kg_protocol_b_{m}")) for r in summary]
        )
        core_metric_avg[f"best_single_excl_naive_{m}"] = mean_or_none(
            [safe_float(r.get(f"best_single_excl_naive_{m}")) for r in summary]
        )
    robust_metric_avg = {}
    for m in ROBUST_METRICS:
        robust_metric_avg[f"kg_protocol_b_{m}"] = mean_or_none(
            [safe_float(r.get(f"kg_protocol_b_{m}")) for r in summary]
        )
        robust_metric_avg[f"best_single_excl_naive_{m}"] = mean_or_none(
            [safe_float(r.get(f"best_single_excl_naive_{m}")) for r in summary]
        )

    slice_metric_avg: Dict[str, Dict[str, Optional[float]]] = {}
    for slice_key in SLICE_KEYS:
        slice_metric_avg[slice_key] = {
            "best_single_excl_mae": mean_or_none(
                [safe_float(r.get(f"best_single_excl_{slice_key}_mae")) for r in summary]
            ),
            "best_single_excl_rmse": mean_or_none(
                [safe_float(r.get(f"best_single_excl_{slice_key}_rmse")) for r in summary]
            ),
            "kg_component_best_mae": mean_or_none(
                [safe_float(r.get(f"kg_component_best_{slice_key}_mae")) for r in summary]
            ),
            "kg_component_best_rmse": mean_or_none(
                [safe_float(r.get(f"kg_component_best_{slice_key}_rmse")) for r in summary]
            ),
        }
    temporal_slice_avg = {
        s: mean_or_none([safe_float(r.get(f"kg_protocol_b_temporal_{s}_mae")) for r in summary])
        for s in ["head", "mid", "tail"]
    }

    report_lines.extend(["", "## KG vs Baseline 改进", ""])
    if improvements:
        avg_improvement = sum(i["improvement_pct"] for i in improvements) / len(improvements)
        positive_count = sum(1 for i in improvements if i["improvement_pct"] > 0)
        report_lines.append(f"- 平均改进: {avg_improvement:.2f}%")
        report_lines.append(f"- KG 更优任务数: {positive_count}/{len(improvements)}")
        report_lines.append(
            f"- catastrophic rate (< -3%): {catastrophic_lt_3}/{len(improvements)}"
        )
        report_lines.append(
            f"- catastrophic rate (< -5%): {catastrophic_lt_5}/{len(improvements)}"
        )

    report_lines.extend(["", "## 搜索噪声与复杂度", ""])
    if candidate_counts:
        report_lines.append(
            f"- 平均候选模型数（common_models_count）: {sum(candidate_counts)/len(candidate_counts):.2f}"
        )
    if common_counts:
        report_lines.append(
            f"- 平均可比较模型数（common）: {sum(common_counts)/len(common_counts):.2f}"
        )
    if safe_counts:
        report_lines.append(
            f"- 平均 safe_models 数: {sum(safe_counts)/len(safe_counts):.2f}"
        )
    if runtime_pairs:
        ratios = [
            (b / a) for a, b in runtime_pairs if a > 1e-12 and b >= 0.0
        ]
        if ratios:
            report_lines.append(
                f"- Protocol B / A 平均时延倍率: {sum(ratios)/len(ratios):.2f}x"
            )
    if pool_mode_counts:
        mode_desc = ", ".join(
            [f"{k}={v}" for k, v in sorted(pool_mode_counts.items(), key=lambda kv: kv[0])]
        )
        report_lines.append(f"- pool_mode 分布: {mode_desc}")

    report_lines.extend(["", "## KG vs Best Single（三口径）", ""])
    ks_eligible: List[float] = []
    ks_excl: List[float] = []
    ks_incl: List[float] = []
    ks_eligible_details: List[Dict[str, Any]] = []
    ks_excl_details: List[Dict[str, Any]] = []
    for row in summary:
        kg_best = None
        for s in ["kg_protocol_a", "kg_protocol_b"]:
            val = row.get(s)
            if val is not None and (kg_best is None or val < kg_best):
                kg_best = val
        bs_eligible = row.get("best_eligible_single_mae")
        bs_excl = row.get("best_single_excl_naive_mae")
        bs_incl = row.get("best_single_incl_naive_mae")
        if kg_best is not None and bs_eligible is not None:
            gap_pct_val = (kg_best - bs_eligible) / (bs_eligible + 1e-10) * 100
            ks_eligible.append(gap_pct_val)
            ks_eligible_details.append({
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                "task": build_task_id(row["dataset"], row["horizon"]),
                "kg_best": float(kg_best),
                "best_eligible_single_mae": float(bs_eligible),
                "gap_pct": float(gap_pct_val),
            })
        if kg_best is not None and bs_excl is not None:
            gap_pct_val = (kg_best - bs_excl) / (bs_excl + 1e-10) * 100
            ks_excl.append(gap_pct_val)
            ks_excl_details.append({
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                "task": build_task_id(row["dataset"], row["horizon"]),
                "kg_best": float(kg_best),
                "best_single_excl_naive_mae": float(bs_excl),
                "gap_pct": float(gap_pct_val),
            })
        if kg_best is not None and bs_incl is not None:
            ks_incl.append((kg_best - bs_incl) / (bs_incl + 1e-10) * 100)

    worst_case_degradation_vs_eligible = None
    if ks_eligible_details:
        worst_case_degradation_vs_eligible = max(ks_eligible_details, key=lambda x: x["gap_pct"])
    worst_case_degradation_vs_excl = None
    if ks_excl_details:
        worst_case_degradation_vs_excl = max(ks_excl_details, key=lambda x: x["gap_pct"])
    catastrophic_vs_eligible = {
        "gt_3pct": sum(1 for x in ks_eligible if x > 3.0),
        "gt_5pct": sum(1 for x in ks_eligible if x > 5.0),
        "total_tasks": len(ks_eligible),
    }
    catastrophic_vs_excl = {
        "gt_3pct": sum(1 for x in ks_excl if x > 3.0),
        "gt_5pct": sum(1 for x in ks_excl if x > 5.0),
        "total_tasks": len(ks_excl),
    }
    if ks_eligible:
        report_lines.append(
            f"- Eligible single: 平均差距 {sum(ks_eligible)/len(ks_eligible):.2f}% (负值=KG更优), "
            f"KG 更优 {sum(1 for x in ks_eligible if x < 0)}/{len(ks_eligible)}"
        )
    if ks_excl:
        report_lines.append(
            f"- 不含 seasonal_naive: 平均差距 {sum(ks_excl)/len(ks_excl):.2f}% (负值=KG更优), "
            f"KG 更优 {sum(1 for x in ks_excl if x < 0)}/{len(ks_excl)}"
        )
    if ks_incl:
        report_lines.append(
            f"- 含 seasonal_naive: 平均差距 {sum(ks_incl)/len(ks_incl):.2f}% (负值=KG更优), "
            f"KG 更优 {sum(1 for x in ks_incl if x < 0)}/{len(ks_incl)}"
        )

    report_lines.extend(["", "## 核心指标均值（KG-B vs Best Single Excl Naive）", ""])
    report_lines.extend([
        "| Metric | KG-B Avg | Best Single Avg | Gap (KG-B vs Single) |",
        "|--------|---------:|----------------:|----------------------:|",
    ])
    for metric_name in CORE_METRICS + ROBUST_METRICS:
        kg_avg = core_metric_avg.get(f"kg_protocol_b_{metric_name}")
        bs_avg = core_metric_avg.get(f"best_single_excl_naive_{metric_name}")
        if metric_name in ROBUST_METRICS:
            kg_avg = robust_metric_avg.get(f"kg_protocol_b_{metric_name}")
            bs_avg = robust_metric_avg.get(f"best_single_excl_naive_{metric_name}")
        gap = gap_pct(bs_avg, kg_avg)
        report_lines.append(
            f"| {metric_name} | {safe_format(kg_avg)} | {safe_format(bs_avg)} | {safe_pct(gap)} |"
        )

    report_lines.extend(["", "## Worst-case 与翻车率（相对 Best Single）", ""])
    if worst_case_degradation_vs_eligible is not None:
        report_lines.append(
            "- worst-case degradation vs best_eligible_single: "
            f"{worst_case_degradation_vs_eligible['gap_pct']:.2f}% "
            f"at {worst_case_degradation_vs_eligible['task']} "
            f"(KG={worst_case_degradation_vs_eligible['kg_best']:.3f}, "
            f"Single={worst_case_degradation_vs_eligible['best_eligible_single_mae']:.3f})"
        )
    if worst_case_degradation_vs_excl is not None:
        report_lines.append(
            "- worst-case degradation vs best_single_excl_naive: "
            f"{worst_case_degradation_vs_excl['gap_pct']:.2f}% "
            f"at {worst_case_degradation_vs_excl['task']} "
            f"(KG={worst_case_degradation_vs_excl['kg_best']:.3f}, "
            f"Single={worst_case_degradation_vs_excl['best_single_excl_naive_mae']:.3f})"
        )
    if catastrophic_vs_eligible["total_tasks"] > 0:
        report_lines.append(
            "- catastrophic rate vs best_eligible_single: "
            f">3% = {catastrophic_vs_eligible['gt_3pct']}/{catastrophic_vs_eligible['total_tasks']}, "
            f">5% = {catastrophic_vs_eligible['gt_5pct']}/{catastrophic_vs_eligible['total_tasks']}"
        )
    if catastrophic_vs_excl["total_tasks"] > 0:
        report_lines.append(
            "- catastrophic rate vs best_single_excl_naive: "
            f">3% = {catastrophic_vs_excl['gt_3pct']}/{catastrophic_vs_excl['total_tasks']}, "
            f">5% = {catastrophic_vs_excl['gt_5pct']}/{catastrophic_vs_excl['total_tasks']}"
        )

    report_lines.extend(["", "## Protocol B 参与率与回退", ""])
    if summary:
        report_lines.append(
            f"- B engaged rate: {protocol_b_engaged_count}/{len(summary)} "
            f"({(100.0 * protocol_b_engaged_count / len(summary)):.1f}%)"
        )
        report_lines.append(
            f"- B fallback count: {protocol_b_fallback_count}/{len(summary)}"
        )
    if fallback_target_counts:
        tgt_desc = ", ".join(
            [f"{k}={v}" for k, v in sorted(fallback_target_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        )
        report_lines.append(f"- fallback target topk: {tgt_desc}")
    if guard_reason_counts:
        reason_desc = ", ".join(
            [f"{k}={v}" for k, v in sorted(guard_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        )
        report_lines.append(f"- fallback reason topk: {reason_desc}")

    report_lines.extend(["", "## 切片指标（可解释性）", ""])
    report_lines.append("- Best Single (No Naive) 切片 MAE 均值：")
    for slice_key in SLICE_KEYS:
        mae_avg = slice_metric_avg.get(slice_key, {}).get("best_single_excl_mae")
        if mae_avg is not None:
            report_lines.append(f"  - {slice_key}: {mae_avg:.3f}")
    report_lines.append("- KG Component Best 切片 MAE 均值：")
    for slice_key in SLICE_KEYS:
        mae_avg = slice_metric_avg.get(slice_key, {}).get("kg_component_best_mae")
        if mae_avg is not None:
            report_lines.append(f"  - {slice_key}: {mae_avg:.3f}")
    report_lines.append("- KG-B temporal_slices MAE 均值：")
    for slice_key in ["head", "mid", "tail"]:
        v = temporal_slice_avg.get(slice_key)
        if v is not None:
            report_lines.append(f"  - temporal_{slice_key}: {v:.3f}")

    report_lines.extend(["", "## KG Component vs Baseline 差距", ""])
    if kg_component_gaps:
        avg_gap = sum(i["gap_pct"] for i in kg_component_gaps) / len(kg_component_gaps)
        better_count = sum(1 for i in kg_component_gaps if i["gap_pct"] < 0)
        report_lines.append(f"- 平均差距: {avg_gap:.2f}% (正值=KG Component 更差)")
        report_lines.append(f"- KG Component 更优任务数: {better_count}/{len(kg_component_gaps)}")
    else:
        report_lines.append("- 无可比任务（KG Component 或 baseline 缺失）")

    report_lines.extend(["", "## 口径一致性检查", ""])
    if consistency_issues:
        report_lines.append(f"- 发现 {len(consistency_issues)} 个潜在问题")
        for issue in consistency_issues[:20]:
            report_lines.append(
                f"- [{issue['type']}] {issue['dataset']} h={issue['horizon']}: {issue['detail']}"
            )
        if len(consistency_issues) > 20:
            report_lines.append(f"- ... 其余 {len(consistency_issues) - 20} 条见 JSON")
    else:
        report_lines.append("- 未发现明显口径冲突")

    report_lines.extend(["", "## 覆盖率口径", ""])
    if expected_tasks > 0:
        report_lines.append(
            f"- 有效任务覆盖率: {valid_tasks}/{expected_tasks} ({(100.0 * valid_tasks / expected_tasks):.1f}%)"
        )
        report_lines.append(
            f"- KG 有效覆盖率: {kg_valid_tasks}/{expected_tasks} ({(100.0 * kg_valid_tasks / expected_tasks):.1f}%)"
        )
        report_lines.append(
            f"- KG-B 运行成功任务数: {kg_protocol_b_runtime_ok_tasks}/{expected_tasks} "
            f"({(100.0 * kg_protocol_b_runtime_ok_tasks / expected_tasks):.1f}%)"
        )
    else:
        report_lines.append("- expected_tasks=0，无法计算覆盖率")

    report_lines.extend(["", "## 候选审计口径", ""])
    audit_applied_tasks = sum(1 for row in summary if row.get("candidate_audit_applied"))
    if summary:
        report_lines.append(f"- candidate audit 已应用任务数: {audit_applied_tasks}/{len(summary)}")
    accepted_vals = [row.get("candidate_audit_accepted_count") for row in summary if row.get("candidate_audit_accepted_count") is not None]
    removed_vals = [row.get("candidate_audit_removed_count") for row in summary if row.get("candidate_audit_removed_count") is not None]
    soft_vals = [row.get("candidate_audit_soft_pass_count") for row in summary if row.get("candidate_audit_soft_pass_count") is not None]
    removed_unaudited_vals = [
        row.get("candidate_audit_removed_unaudited_count")
        for row in summary
        if row.get("candidate_audit_removed_unaudited_count") is not None
    ]
    if accepted_vals:
        report_lines.append(f"- 平均 accepted 扩展节点数: {sum(accepted_vals)/len(accepted_vals):.2f}")
    if removed_vals:
        report_lines.append(f"- 平均 removed 扩展节点数: {sum(removed_vals)/len(removed_vals):.2f}")
    if soft_vals:
        report_lines.append(f"- 平均 soft-pass 扩展节点数: {sum(soft_vals)/len(soft_vals):.2f}")
    if removed_unaudited_vals:
        report_lines.append(
            f"- 平均 removed unaudited 扩展节点数: {sum(removed_unaudited_vals)/len(removed_unaudited_vals):.2f}"
        )

    report_lines.extend(["", "## KG 运行效率", ""])
    rt_a = [float(r["runtime_protocol_a_sec"]) for r in summary if r.get("runtime_protocol_a_sec") is not None]
    rt_b = [float(r["runtime_protocol_b_sec"]) for r in summary if r.get("runtime_protocol_b_sec") is not None]
    if rt_a:
        report_lines.append(f"- Protocol A 平均运行时长: {sum(rt_a)/len(rt_a):.3f}s")
    if rt_b:
        report_lines.append(f"- Protocol B 平均运行时长: {sum(rt_b)/len(rt_b):.3f}s")
    naive_sel = [
        float(r["naive_selected_ratio_protocol_b"])
        for r in summary
        if r.get("naive_selected_ratio_protocol_b") is not None
    ]
    if naive_sel:
        report_lines.append(f"- seasonal_naive 被 KG-B 选中比例: {sum(naive_sel)/len(naive_sel):.2f}")

    report_lines.extend(["", "## Guard 触发频次", ""])
    if guard_reason_counts:
        report_lines.extend([
            "| Guard | 触发次数 | 涉及 Task |",
            "|-------|---------:|-----------|",
        ])
        for code, count in sorted(guard_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            tasks = guard_reason_tasks.get(code, [])
            task_display = ", ".join(tasks[:4])
            if len(tasks) > 4:
                task_display += f", ...(+{len(tasks) - 4})"
            report_lines.append(f"| {code} | {count} | {task_display or 'N/A'} |")
    else:
        report_lines.append("- 本次运行未触发 Protocol B guard 回退。")

    report_lines.extend(["", "## Interaction 诊断", ""])
    if interaction_disabled_counts:
        for code, cnt in sorted(interaction_disabled_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            report_lines.append(f"- disabled_reason `{code}`: {cnt} 次")
    else:
        report_lines.append("- 无 interaction disabled 记录。")
    if interaction_reject_counts:
        for code, cnt in sorted(interaction_reject_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            report_lines.append(f"- reject_reason `{code}`: {cnt} 次")

    b_runtime_errors = [r for r in summary if r.get("kg_protocol_b_error")]
    if b_runtime_errors:
        report_lines.extend(["", "## Protocol B 运行异常", ""])
        for r in b_runtime_errors:
            report_lines.append(
                f"- {r['dataset']} h={r['horizon']}: `{r.get('kg_protocol_b_error')}`"
            )

    ensure_dir(out_path.parent)
    report_text = "\n".join(report_lines)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    json_out = out_path.with_suffix(".json")
    with json_out.open("w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "win_counts": win_counts,
            "improvements": improvements,
            "kg_vs_best_single_eligible": ks_eligible,
            "kg_vs_best_single_excl_naive": ks_excl,
            "kg_vs_best_single_incl_naive": ks_incl,
            "kg_component_gaps": kg_component_gaps,
            "consistency_issues": consistency_issues,
            "coverage": {
                "expected_tasks": expected_tasks,
                "valid_tasks": valid_tasks,
                "kg_valid_tasks": kg_valid_tasks,
                "kg_protocol_b_runtime_ok_tasks": kg_protocol_b_runtime_ok_tasks,
                "candidate_audit_applied_tasks": audit_applied_tasks,
                "valid_coverage_pct": (100.0 * valid_tasks / expected_tasks) if expected_tasks else None,
                "kg_coverage_pct": (100.0 * kg_valid_tasks / expected_tasks) if expected_tasks else None,
            },
            "protocol_b_guard_stats": {
                "counts": guard_reason_counts,
                "tasks": guard_reason_tasks,
            },
            "interaction_stats": {
                "disabled_counts": interaction_disabled_counts,
                "reject_counts": interaction_reject_counts,
            },
            "pool_mode_counts": pool_mode_counts,
            "catastrophic_rate": {
                "lt_minus_3pct": catastrophic_lt_3,
                "lt_minus_5pct": catastrophic_lt_5,
                "total_tasks": len(improvements),
            },
            "catastrophic_rate_vs_best_single": {
                "vs_best_eligible_single": catastrophic_vs_eligible,
                "vs_best_single_excl_naive": catastrophic_vs_excl,
            },
            "worst_case_degradation": {
                "vs_best_eligible_single": worst_case_degradation_vs_eligible,
                "vs_best_single_excl_naive": worst_case_degradation_vs_excl,
            },
            "core_metric_averages": core_metric_avg,
            "robust_metric_averages": robust_metric_avg,
            "slice_metric_averages": slice_metric_avg,
            "temporal_slice_averages_kg_b": temporal_slice_avg,
            "protocol_b_engagement": {
                "engaged_count": protocol_b_engaged_count,
                "total_tasks": len(summary),
                "engaged_rate": (protocol_b_engaged_count / len(summary)) if summary else None,
                "fallback_count": protocol_b_fallback_count,
            },
            "fallback_stats": {
                "target_counts": fallback_target_counts,
                "reason_counts": guard_reason_counts,
            },
            "kg_vs_best_single_details": {
                "eligible": ks_eligible_details,
                "excl_naive": ks_excl_details,
            },
            "search_noise": {
                "avg_candidate_models_count": (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else None,
                "avg_common_models_count": (sum(common_counts) / len(common_counts)) if common_counts else None,
                "avg_safe_models_count": (sum(safe_counts) / len(safe_counts)) if safe_counts else None,
            },
            "sources": {
                "baseline_root": str(baseline_root),
                "kg_root": str(kg_root),
                "leaderboard_path": str(leaderboard_path) if leaderboard_path else None,
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"报告已保存到: {out_path}")
    print(f"JSON 数据已保存到: {json_out}")
    return summary, win_counts


def main():
    parser = argparse.ArgumentParser(description="统一对比 Baseline / KG Core / KG Component 结果")
    parser.add_argument("--baseline-root", type=Path, default=Path("reports/combos_fixed"),
                        help="Baseline 结果目录")
    parser.add_argument("--kg-root", type=Path, default=Path("reports/combos_kg"),
                        help="KG 结果目录")
    parser.add_argument("--leaderboard-root", type=Path, default=None,
                        help="可选：包含 leaderboard_full.csv 的目录（通常是 modelcombine 输出目录）")
    parser.add_argument("--out", type=Path, default=Path("reports/analysis/comparison_report.md"),
                        help="输出报告路径")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="仅报告指定数据集（默认: pjm aemo_vic aemo_nsw）")
    # 旧参数兼容：避免历史脚本因参数名差异直接报错
    parser.add_argument("--baseline_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--kg_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--kg_component_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    baseline_arg = args.baseline_dir if args.baseline_dir is not None else args.baseline_root
    kg_arg = args.kg_dir if args.kg_dir is not None else args.kg_root
    leaderboard_arg = args.kg_component_dir if args.kg_component_dir is not None else args.leaderboard_root
    out_arg = args.output if args.output is not None else args.out

    baseline_root = baseline_arg if baseline_arg.is_absolute() else project_root / baseline_arg
    kg_root = kg_arg if kg_arg.is_absolute() else project_root / kg_arg
    leaderboard_root = None
    if leaderboard_arg is not None:
        leaderboard_root = (
            leaderboard_arg if leaderboard_arg.is_absolute()
            else project_root / leaderboard_arg
        )
    out_path = out_arg if out_arg.is_absolute() else project_root / out_arg

    generate_comparison_report(
        baseline_root=baseline_root,
        kg_root=kg_root,
        out_path=out_path,
        leaderboard_root=leaderboard_root,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
