"""System A/B 影子基准：只读测量与根因定位（合一计划 Task 5）。

本脚本**只测量，不修改任何算法、guard 阈值或数据契约**。它：

- 在隔离临时目录中运行，绝不改写仓库 `reports/`、历史场景库或生产图谱
  （运行前后对这些文件做哈希核对，不一致即判失败）；
- 分阶段记录耗时，并把模型训练耗时与 Protocol B solver 耗时**分开**统计；
- 用 `cProfile` 给出 top cumulative 调用，作为"慢在哪个调用"的直接证据；
- 单次运行超时时保存已完成阶段、异常位置与 profiler top calls，然后以非零
  状态退出——不允许只写一句"运行较慢"。

用法：
    python scripts/profile_system_ab_demo.py --rows 168 --repeat 2 \
        --output result/ab_convergence/profile_168.json
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

REPORT_SCHEMA_VERSION = "task5.1"
DEFAULT_TIMEOUT_SECONDS = 300.0

# 运行前后必须保持不变的生产状态文件（只读保证）
GUARDED_PATHS = [
    "reports/historical_scenarios.json",
    "reports/graph_state.pkl",
    "reports/predictions.csv",
]

# 阶段名 -> 被包装的函数；用于把耗时归因到计划要求的各阶段。
STAGE_ORDER = [
    "data_and_features",
    "candidate_training_round1_fit",
    "candidate_training_round2_full",
    "prediction_bundle",
    "protocol_a",
    "stability_filter",
    "feature_corr_and_graph",
    "blocked_cv",
    "ridge_weight_fit",
    "result_assembly",
    "interaction_post_adjust_guard_residual",
    "protocol_b_solver_total",
]


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "<absent>"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _snapshot_guarded(root: Path) -> Dict[str, str]:
    return {rel: _sha256_file(root / rel) for rel in GUARDED_PATHS}


class StageTimer:
    """累计各阶段耗时；仅包装函数，不改变其行为与返回值。"""

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._patches: List[tuple] = []

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + float(seconds)
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def wrap(self, module: Any, name: str, stage: str) -> None:
        original = getattr(module, name, None)
        if original is None:
            return

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.add(stage, time.perf_counter() - t0)

        setattr(module, name, timed)
        self._patches.append((module, name, original))

    def restore(self) -> None:
        for module, name, original in reversed(self._patches):
            setattr(module, name, original)
        self._patches.clear()


def _install_stage_wrappers(timer: StageTimer) -> None:
    """包装各阶段代表函数。函数位置以 protocol_b.py 的真实导入为准。"""
    import src.eval.kg.protocol_b as pb
    import src.pipeline.prediction_pool as pool

    timer.wrap(pb, "kg_combination_pred_only", "protocol_a")
    timer.wrap(pb, "_dedup_and_stability_filter", "stability_filter")
    timer.wrap(pb, "compute_feature_model_correlation_safe", "feature_corr_and_graph")
    timer.wrap(pb, "blocked_cv_select_alpha", "blocked_cv")
    timer.wrap(pb, "blocked_cv_splits", "blocked_cv")
    timer.wrap(pb, "_blocked_cv_mae_from_pred", "blocked_cv")
    timer.wrap(pb, "_interaction_oof_cv_metrics", "blocked_cv")
    timer.wrap(pb, "fit_static_weight_ridge", "ridge_weight_fit")
    timer.wrap(pb, "_cleanup_zero_weight_models_and_refit", "ridge_weight_fit")
    timer.wrap(pb, "_merge_eval_metrics", "result_assembly")
    timer.wrap(pool, "build_matrix", "prediction_bundle")


class _TimingModelProxy:
    """把候选模型的 fit/predict 计入对应训练轮次，且不改变数值行为。"""

    def __init__(self, inner: Any, timer: StageTimer, round_state: Dict[str, str],
                 deadline: Optional[float] = None, model_key: str = "?",
                 per_model: Optional[Dict[str, Dict[str, float]]] = None):
        self._inner = inner
        self._timer = timer
        self._round_state = round_state
        self._deadline = deadline
        self._model_key = model_key
        self._per_model = per_model if per_model is not None else {}

    def fit(self, X, y):
        # 超时检查必须发生在训练**之前**：full 档单个深度模型可能训练很久，
        # 只在阶段之间检查会导致脚本永远走不到检查点。
        if self._deadline is not None and time.perf_counter() > self._deadline:
            raise TimeoutError(
                f"deadline exceeded before fitting {type(self._inner).__name__} "
                f"in stage {self._round_state['stage']}"
            )
        t0 = time.perf_counter()
        try:
            return self._inner.fit(X, y)
        finally:
            dt = time.perf_counter() - t0
            self._timer.add(self._round_state["stage"], dt)
            # 逐模型训练耗时：用于回答"慢在哪个调用"，而不是只说"训练慢"。
            slot = self._per_model.setdefault(self._model_key, {})
            slot[self._round_state["stage"]] = round(
                slot.get(self._round_state["stage"], 0.0) + dt, 4
            )

    def predict(self, X):
        t0 = time.perf_counter()
        try:
            return self._inner.predict(X)
        finally:
            self._timer.add(self._round_state["stage"], time.perf_counter() - t0)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _load_demo_frame(rows: str) -> pd.DataFrame:
    """只读加载 demo 数据并构建特征（与 run.py 同一套特征函数）。"""
    from src.features.build_features import (
        add_holiday_feature,
        add_lag_rolling,
        add_time_features,
        join_weather,
    )
    from src.utils.io import load_yaml

    cfg = load_yaml(os.path.join(PROJECT_ROOT, "configs", "pipeline.yaml"))
    root = os.path.join(PROJECT_ROOT, cfg["data"]["root"])
    load_df = pd.read_csv(os.path.join(root, "load.csv"))
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"])
    load_df = load_df.sort_values(["region", "timestamp"])

    # 特征含 lag_168 / roll168，dropna 会吃掉开头约 168 行。若直接切 N 行，
    # 168 档会被清空。因此先多取一段缓冲，构完特征再取尾部 N 行可用样本。
    warmup = 0
    if rows != "full":
        warmup = 200
        load_df = load_df.groupby("region", group_keys=False).tail(int(rows) + warmup)

    weather_path = os.path.join(root, "weather.csv")
    if os.path.exists(weather_path):
        load_df = join_weather(load_df, pd.read_csv(weather_path))
    load_df = add_time_features(load_df)
    load_df = add_holiday_feature(load_df)
    load_df = add_lag_rolling(
        load_df,
        cfg["features"]["lags"],
        cfg["features"]["rolling"],
    )
    load_df = load_df.dropna()
    if rows != "full":
        load_df = load_df.groupby("region", group_keys=False).tail(int(rows))
    return load_df.reset_index(drop=True)


def _collect_guard_evidence(raw: Dict[str, Any], model_cols: List[str],
                            df_val: pd.DataFrame) -> Dict[str, Any]:
    """逐模型抽取 guard 移除规则、输入统计量与阈值（只读，不改行为）。"""
    split = raw.get("test") or raw.get("val") or {}
    weight_meta = split.get("weight_meta") or {}
    selection_meta = weight_meta.get("protocol_b_selection_meta") or {}
    stability = selection_meta.get("stability") or {}
    guard = weight_meta.get("protocol_b_guard") or {}
    guard_config = weight_meta.get("guard_config") or {}

    per_model = {}
    y_val = np.asarray(df_val["y"].values, dtype=float)
    by_model_meta = stability.get("by_model") or {}
    for m in model_cols:
        col = np.asarray(df_val[m].values, dtype=float) if m in df_val.columns else np.array([])
        finite = col[np.isfinite(col)] if col.size else col
        per_model[m] = {
            "removed_by_stability": m in (stability.get("removed_models") or {}),
            "stability_meta": by_model_meta.get(m),
            "val_mae": (
                float(np.mean(np.abs(col - y_val))) if col.size == y_val.size and col.size else None
            ),
            "val_pred_std": float(np.std(finite)) if finite.size else None,
            "val_pred_nan_ratio": (
                float(1.0 - finite.size / col.size) if col.size else None
            ),
        }

    return {
        "final_selected_models": split.get("selected_models"),
        "protocol_b_candidates": split.get("selected_models_b_candidate"),
        "stability_removed_models": stability.get("removed_models"),
        "fallback_target": guard.get("fallback_target") or guard_config.get("final_fallback_target"),
        "fallback_reason": guard.get("reason") or guard_config.get("final_fallback_reason"),
        "guard_thresholds": {
            k: v for k, v in guard_config.items()
            if not isinstance(v, (dict, list))
        },
        "protocol_b_guard_metrics": {
            k: v for k, v in guard.items() if not isinstance(v, (dict, list))
        },
        "per_model": per_model,
    }


def _run_once(rows: str, run_index: int, timeout_seconds: float) -> Dict[str, Any]:
    import src.pipeline.prediction_pool as pool
    from src.models.registry import model_registry
    from src.pipeline.protocol_b_adapter import DemoProtocolBAdapter

    timer = StageTimer()
    record: Dict[str, Any] = {
        "run_index": run_index,
        "rows": rows,
        "status": "unknown",
        "stages_completed": [],
        "error": None,
        "error_stage": None,
    }

    t_start = time.perf_counter()
    profiler = cProfile.Profile()
    round_state = {"stage": "candidate_training_round1_fit"}
    original_create = model_registry.create

    try:
        t0 = time.perf_counter()
        df = _load_demo_frame(rows)
        timer.add("data_and_features", time.perf_counter() - t0)
        record["stages_completed"].append("data_and_features")

        region = str(df["region"].iloc[0])
        region_df = df[df["region"] == region].copy()
        n = len(region_df)
        test_n = max(24, int(n * 0.1))
        train, test = region_df.iloc[: n - test_n], region_df.iloc[n - test_n :]
        record["region"] = region
        record["n_rows_region"] = int(n)
        record["n_train"] = int(len(train))
        record["n_test"] = int(len(test))
        record["input_data_sha"] = hashlib.sha256(
            pd.util.hash_pandas_object(region_df, index=True).values.tobytes()
        ).hexdigest()[:16]

        _install_stage_wrappers(timer)

        # 两轮训练分别计时：bundle 内部先用 fit 段、再用完整 train 重训。
        deadline = t_start + timeout_seconds

        per_model_seconds: Dict[str, Dict[str, float]] = {}

        def counting_create(key, **params):
            return _TimingModelProxy(
                original_create(key, **params), timer, round_state,
                deadline=deadline, model_key=key, per_model=per_model_seconds,
            )

        model_registry.create = counting_create
        pool_registry = _RoundAwareRegistry(model_registry, round_state)

        candidate_models = [
            m for m in model_registry.get_available_models() if "blender" not in m.lower()
        ]
        record["candidate_models"] = candidate_models

        # 观察到的契约约束：配置默认 validation_days=30，但小窗口（如 168 行=7 天）
        # 的训练段根本切不出 30 天验证窗，split_fit_validation 会直接报错。
        # 基准脚本按可用跨度自适应，并记录实际取值，便于 Task 6 判断是配置问题
        # 还是契约问题（本脚本不改任何生产默认值）。
        span_days = max(
            1.0,
            (pd.to_datetime(train["timestamp"]).max()
             - pd.to_datetime(train["timestamp"]).min()).total_seconds() / 86400.0,
        )
        validation_days = max(1, min(30, int(span_days * 0.2)))
        record["train_span_days"] = round(span_days, 2)
        record["validation_days_used"] = validation_days
        record["validation_days_config_default"] = 30

        t0 = time.perf_counter()
        bundle = pool.build_region_prediction_bundle(
            region=region,
            train=train,
            test=test,
            candidate_models=candidate_models,
            validation_days=validation_days,
            registry=pool_registry,
        )
        bundle_wall = time.perf_counter() - t0
        timer.add("prediction_bundle", 0.0)  # 确保键存在
        record["stages_completed"].append("prediction_bundle")
        record["bundle_wall_seconds"] = round(bundle_wall, 4)
        record["per_model_training_seconds"] = per_model_seconds
        record["surviving_model_cols"] = list(bundle.model_cols)
        record["failed_models"] = bundle.metadata.get("failed_models")

        if time.perf_counter() - t_start > timeout_seconds:
            raise TimeoutError(
                f"exceeded {timeout_seconds}s after prediction_bundle "
                f"({time.perf_counter() - t_start:.1f}s elapsed)"
            )

        # Protocol B solver 计时：不含上面的候选模型训练
        t0 = time.perf_counter()
        profiler.enable()
        adapter_result = DemoProtocolBAdapter().select(bundle, region=region, horizon=1)
        profiler.disable()
        solver_wall = time.perf_counter() - t0
        timer.add("protocol_b_solver_total", solver_wall)
        record["stages_completed"].append("protocol_b_solver")

        raw = adapter_result.get("raw") or {}
        record["protocol"] = adapter_result.get("strategy")
        record["yhat_source"] = adapter_result.get("yhat_source")
        record["protocol_b"] = {
            "selected_models": adapter_result.get("models"),
            "weights": adapter_result.get("weights"),
            "mae": adapter_result.get("mae"),
            "linear_reconstruction_match": adapter_result.get("linear_reconstruction_match"),
        }
        record["guard_evidence"] = _collect_guard_evidence(
            raw, list(bundle.model_cols), bundle.df_val
        )
        record["status"] = "ok"

    except BaseException as exc:  # noqa: BLE001 - 需要在超时/异常时仍保存证据
        profiler.disable()
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_stage"] = (
            record["stages_completed"][-1] if record["stages_completed"] else "data_and_features"
        )
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-6:]
        try:
            record.setdefault("per_model_training_seconds", per_model_seconds)
        except NameError:
            pass
    finally:
        model_registry.create = original_create
        timer.restore()

    record["wall_seconds"] = round(time.perf_counter() - t_start, 4)
    stages = {k: round(v, 4) for k, v in timer.totals.items()}
    measured = sum(
        stages.get(k, 0.0)
        for k in ("protocol_a", "stability_filter", "feature_corr_and_graph",
                  "blocked_cv", "ridge_weight_fit", "result_assembly")
    )
    stages["interaction_post_adjust_guard_residual"] = round(
        max(0.0, stages.get("protocol_b_solver_total", 0.0) - measured), 4
    )
    record["stage_seconds"] = {k: stages.get(k, 0.0) for k in STAGE_ORDER}
    record["stage_call_counts"] = dict(timer.counts)
    record["profiler_top_calls"] = _top_calls(profiler)
    return record


class _RoundAwareRegistry:
    """在 bundle 的第二轮（完整 train 重训）切换计时归属。"""

    def __init__(self, inner: Any, round_state: Dict[str, str]):
        self._inner = inner
        self._round_state = round_state
        self._seen = set()

    def create(self, key: str, **params):
        if key in self._seen:
            self._round_state["stage"] = "candidate_training_round2_full"
        else:
            self._seen.add(key)
            self._round_state["stage"] = "candidate_training_round1_fit"
        return self._inner.create(key, **params)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _top_calls(profiler: cProfile.Profile, limit: int = 15) -> List[Dict[str, Any]]:
    try:
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf)
        stats.sort_stats("cumulative")
    except (TypeError, ValueError):
        return []
    rows: List[Dict[str, Any]] = []
    for func, (cc, nc, tt, ct, _callers) in stats.stats.items():
        rows.append({
            "func": f"{Path(func[0]).name}:{func[1]}({func[2]})",
            "ncalls": int(nc),
            "tottime": round(tt, 4),
            "cumtime": round(ct, 4),
        })
    rows.sort(key=lambda r: r["cumtime"], reverse=True)
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="System A/B 影子基准（只读）")
    parser.add_argument("--rows", default="168", help="168 | 720 | full")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    repo = Path(PROJECT_ROOT)
    before = _snapshot_guarded(repo)

    # 隔离：adapter 的 trace 与任何落盘都指向临时目录
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="ab_profile_")
    import src.pipeline.protocol_b_adapter as adapter_mod
    import src.pipeline.main as pipeline_main

    pipeline_main.PROJECT_ROOT = tmpdir

    runs: List[Dict[str, Any]] = []
    for i in range(1, max(1, args.repeat) + 1):
        buf = io.StringIO()
        with redirect_stdout(buf):
            runs.append(_run_once(args.rows, i, args.timeout_seconds))

    after = _snapshot_guarded(repo)
    readonly_ok = before == after

    ok_runs = [r for r in runs if r["status"] == "ok"]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "rows": args.rows,
        "repeat": args.repeat,
        "timeout_seconds": args.timeout_seconds,
        "random_seed": None,
        "isolated_tmpdir": tmpdir,
        "guarded_state_before": before,
        "guarded_state_after": after,
        "readonly_guarantee_held": readonly_ok,
        "runs": runs,
        "summary": {
            "n_ok": len(ok_runs),
            "n_failed": len(runs) - len(ok_runs),
            "cold_wall_seconds": runs[0]["wall_seconds"] if runs else None,
            "repeat_wall_seconds": runs[1]["wall_seconds"] if len(runs) > 1 else None,
            "cold_vs_repeat_delta": (
                round(runs[0]["wall_seconds"] - runs[1]["wall_seconds"], 4)
                if len(runs) > 1 else None
            ),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"[profile] rows={args.rows} -> {out} ({size_kb:.1f} KB)")
    print(f"[profile] readonly_guarantee_held={readonly_ok} ok={len(ok_runs)}/{len(runs)}")
    for r in runs:
        print(f"  run{r['run_index']}: status={r['status']} wall={r['wall_seconds']}s "
              f"protocol={r.get('protocol')} err={r.get('error')}")

    if not readonly_ok:
        print("[profile][FATAL] 生产状态在运行前后发生变化，只读保证被破坏")
        return 2
    if len(ok_runs) < len(runs):
        print("[profile][FAIL] 存在失败或超时的运行；已保存已完成阶段与 profiler top calls")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
