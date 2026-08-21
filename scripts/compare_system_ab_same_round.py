"""Task 6A：System A/B 同轮质量对照（只诊断，不修改产品规则）。

本脚本回答三个尚未确认的问题：

1. 在同一份 demo 数据、同一时间切分和隔离初始状态下，Protocol B 是否仍比
   legacy combinator 差；
2. Protocol B 是否同时达到 combinator 与“按 validation 选择的最佳单模型”
   1% 容忍门槛；
3. 若质量差距存在，它是否与深度候选池或 `unstable_late` 过滤有关。

脚本不修改 guard、候选池默认值或 validation_days。传统候选池与
`unstable_late` 关闭均只在内存中做诊断重放，绝不进入产品配置。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.kg.config import RUNTIME_PREDICTIONS_KEY
from src.pipeline.prediction_pool import RegionPredictionBundle, build_region_prediction_bundle

REPORT_SCHEMA_VERSION = "task6a.1"
RANDOM_SEED = 42
QUALITY_TOLERANCE_RATIO = 1.01
DEEP_MODEL_IDS = frozenset({"informer", "autoformer", "powergpt"})


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """所有比较对象共用同一个 MAE/RMSE 定义。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"metric shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def best_single_summaries(bundle: RegionPredictionBundle) -> Dict[str, Any]:
    """同时报告可部署的 validation 选择与只作诊断的 test oracle。"""
    if not bundle.model_cols:
        raise ValueError("best_single_summaries requires at least one model")
    y_val = bundle.df_val["y"].to_numpy(dtype=float)
    y_test = bundle.df_test["y"].to_numpy(dtype=float)
    val_mae = {
        m: float(np.mean(np.abs(bundle.df_val[m].to_numpy(dtype=float) - y_val)))
        for m in bundle.model_cols
    }
    test_metrics = {
        m: metric_summary(y_test, bundle.df_test[m].to_numpy(dtype=float))
        for m in bundle.model_cols
    }
    validation_selected = min(bundle.model_cols, key=lambda m: (val_mae[m], m))
    test_oracle = min(bundle.model_cols, key=lambda m: (test_metrics[m]["mae"], m))
    return {
        "validation_selected": {
            "model": validation_selected,
            "validation_mae": val_mae[validation_selected],
            "test": test_metrics[validation_selected],
            "selection_uses_test_labels": False,
        },
        "test_oracle": {
            "model": test_oracle,
            "validation_mae": val_mae[test_oracle],
            "test": test_metrics[test_oracle],
            "selection_uses_test_labels": True,
        },
        "per_model": {
            m: {"validation_mae": val_mae[m], "test": test_metrics[m]}
            for m in bundle.model_cols
        },
    }


def traditional_model_cols(model_cols: Sequence[str]) -> List[str]:
    """诊断池：只排除 Task 4 新引入的三个深度模型。"""
    return [m for m in model_cols if m not in DEEP_MODEL_IDS]


def subset_bundle(
    bundle: RegionPredictionBundle,
    model_cols: Sequence[str],
) -> RegionPredictionBundle:
    """从同一预测矩阵取候选子集，不重训、不改变任何预测值。"""
    requested = list(model_cols)
    missing = [m for m in requested if m not in bundle.model_cols]
    if missing:
        raise ValueError(f"subset_bundle models not in source bundle: {missing}")
    if not requested:
        raise ValueError("subset_bundle requires at least one model")
    fixed = ["timestamp", "y"]
    return RegionPredictionBundle(
        df_val=bundle.df_val[fixed + requested].copy(),
        df_test=bundle.df_test[fixed + requested].copy(),
        df_raw_val=bundle.df_raw_val.copy(),
        df_raw_test=bundle.df_raw_test.copy(),
        model_cols=requested,
        base_model_cols=[m for m in bundle.base_model_cols if m in requested],
        fitted_test_models={m: bundle.fitted_test_models[m] for m in requested},
        metadata={**bundle.metadata, "diagnostic_subset_models": requested},
    )


def remove_unstable_late_only(
    filter_ctx: Mapping[str, Any],
    *,
    original_model_cols: Sequence[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """纯诊断转换：只恢复唯一原因为 unstable_late 的候选。"""
    updated = copy.deepcopy(dict(filter_ctx))
    removed = dict(updated.get("stability_removed") or {})
    reinstated: List[str] = []
    for model in original_model_cols:
        reason = removed.get(model)
        if reason is None:
            continue
        reasons = {item.strip() for item in str(reason).split(",") if item.strip()}
        if reasons == {"unstable_late"}:
            reinstated.append(model)
            removed.pop(model, None)

    current = list(updated.get("model_cols") or [])
    restored_set = set(current) | set(reinstated)
    updated["model_cols"] = [m for m in original_model_cols if m in restored_set]
    updated["stability_removed"] = removed
    return updated, reinstated


def quality_gate(
    *,
    protocol_b_mae: float,
    combinator_mae: float,
    validation_selected_single_mae: float,
    tolerance_ratio: float = QUALITY_TOLERANCE_RATIO,
) -> Dict[str, Any]:
    """Protocol B 必须同时不劣于两个可部署参考的 1% 容忍线。"""
    def one(reference: float) -> Dict[str, Any]:
        threshold = float(reference) * float(tolerance_ratio)
        return {
            "reference_mae": float(reference),
            "threshold_mae": threshold,
            "ratio": float(protocol_b_mae / reference) if reference else None,
            "passed": bool(protocol_b_mae <= threshold),
        }

    vs_combinator = one(combinator_mae)
    vs_single = one(validation_selected_single_mae)
    return {
        "tolerance_ratio": float(tolerance_ratio),
        "protocol_b_mae": float(protocol_b_mae),
        "vs_combinator": vs_combinator,
        "vs_validation_selected_single": vs_single,
        "passed": bool(vs_combinator["passed"] and vs_single["passed"]),
    }


def validate_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "rows",
        "repeat",
        "python",
        "runs",
        "conclusions",
        "guarded_state_before",
        "guarded_state_after",
        "readonly_guarantee_held",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"same-round report missing keys: {sorted(missing)}")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version: {report['schema_version']}")
    if not report["runs"]:
        raise ValueError("same-round report contains no runs")
    for run in report["runs"]:
        if run.get("status") != "ok":
            raise ValueError(f"same-round run not successful: {run.get('error')}")
        for key in ("input", "combinator", "protocol_a", "protocol_b", "best_single", "quality_gate"):
            if key not in run:
                raise ValueError(f"same-round run missing {key}")
    conclusions = report["conclusions"]
    for key in ("observed_facts", "evidence_supported_inferences", "still_unknown"):
        if key not in conclusions or not isinstance(conclusions[key], list):
            raise ValueError(f"conclusions missing list {key}")


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "<absent>"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _snapshot_production_state() -> Dict[str, str]:
    rels = (
        "reports/historical_scenarios.json",
        "reports/graph_state.pkl",
        "reports/predictions.csv",
        "reports/report.json",
    )
    return {rel: _sha256_file(PROJECT_ROOT / rel) for rel in rels}


def _seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _data_sha(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_and_split(rows: str) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    """full 使用生产 30 天 test；720 使用明确标注的 10% 诊断切分。"""
    from scripts.profile_system_ab_demo import _load_demo_frame
    from src.utils.io import load_yaml

    frame = _load_demo_frame(rows)
    region = str(frame["region"].iloc[0])
    region_df = frame[frame["region"] == region].sort_values("timestamp").copy()
    if rows == "full":
        cfg = load_yaml(str(PROJECT_ROOT / "configs" / "pipeline.yaml"))
        test_days = int(cfg["data"]["test_days"])
        cutoff = pd.to_datetime(region_df["timestamp"]).max() - pd.Timedelta(days=test_days)
        train = region_df[pd.to_datetime(region_df["timestamp"]) <= cutoff].copy()
        test = region_df[pd.to_datetime(region_df["timestamp"]) > cutoff].copy()
        split_meta = {"kind": "production_test_days", "test_days": test_days}
    else:
        test_n = max(24, int(len(region_df) * 0.1))
        train = region_df.iloc[:-test_n].copy()
        test = region_df.iloc[-test_n:].copy()
        split_meta = {"kind": "diagnostic_tail_fraction", "test_rows": test_n}
    if train.empty or test.empty:
        raise ValueError(f"empty train/test after split rows={rows}")
    return train, test, region, split_meta


def _validation_days(train: pd.DataFrame, rows: str) -> int:
    if rows == "full":
        return 30
    span_days = max(
        1.0,
        (
            pd.to_datetime(train["timestamp"]).max()
            - pd.to_datetime(train["timestamp"]).min()
        ).total_seconds()
        / 86400.0,
    )
    return max(1, min(30, int(span_days * 0.2)))


def _protocol_a_summary(bundle: RegionPredictionBundle, region: str) -> Dict[str, Any]:
    from src.eval.kg.protocol_a import kg_combination_pred_only

    t0 = time.perf_counter()
    raw = kg_combination_pred_only(
        bundle.df_val,
        bundle.df_test,
        list(bundle.model_cols),
        1,
        dataset_name=region,
        return_predictions=True,
    )
    elapsed = time.perf_counter() - t0
    runtime_predictions = raw.pop(RUNTIME_PREDICTIONS_KEY)
    yhat = np.asarray(runtime_predictions["test"], dtype=float)
    split = raw.get("test") or {}
    return {
        "protocol": raw.get("protocol"),
        "models": list(split.get("selected_models") or []),
        "weights": dict(split.get("weights") or {}),
        "metrics": metric_summary(bundle.df_test["y"].to_numpy(dtype=float), yhat),
        "elapsed_seconds": elapsed,
    }


def _protocol_b_summary(
    bundle: RegionPredictionBundle,
    region: str,
    *,
    diagnostic_name: str,
) -> Dict[str, Any]:
    from scripts.profile_system_ab_demo import _collect_guard_evidence
    from src.pipeline.protocol_b_adapter import DemoProtocolBAdapter

    t0 = time.perf_counter()
    result = DemoProtocolBAdapter().select(bundle, region=region, horizon=1)
    elapsed = time.perf_counter() - t0
    yhat = np.asarray(result["yhat"], dtype=float)
    return {
        "diagnostic_name": diagnostic_name,
        "protocol": result.get("strategy"),
        "models": list(result.get("models") or []),
        "weights": dict(result.get("weights") or {}),
        "metrics": metric_summary(bundle.df_test["y"].to_numpy(dtype=float), yhat),
        "elapsed_seconds": elapsed,
        "yhat_source": result.get("yhat_source"),
        "linear_reconstruction_match": result.get("linear_reconstruction_match"),
        "guard_evidence": _collect_guard_evidence(
            result.get("raw") or {}, list(bundle.model_cols), bundle.df_val
        ),
    }


@contextmanager
def _diagnostic_without_unstable_late():
    """内存 monkeypatch；退出上下文后完整恢复产品函数。"""
    import src.eval.kg.protocol_b as protocol_b
    from src.eval.kg.conflict import compute_error_correlations

    original = protocol_b._dedup_and_stability_filter

    def diagnostic_filter(*args, **kwargs):
        ctx = original(*args, **kwargs)
        original_models = list(kwargs.get("model_cols") or (args[2] if len(args) > 2 else []))
        updated, reinstated = remove_unstable_late_only(
            ctx, original_model_cols=original_models
        )
        if reinstated:
            updated["error_corrs"] = compute_error_correlations(
                kwargs["df_val"], updated["model_cols"]
            )
            updated["task6a_reinstated_unstable_late"] = reinstated
        return updated

    protocol_b._dedup_and_stability_filter = diagnostic_filter
    try:
        yield
    finally:
        protocol_b._dedup_and_stability_filter = original


def _combinator_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    region: str,
    tmpdir: Path,
) -> Dict[str, Any]:
    """运行 legacy 生产调用链，但不执行 feedback，不写生产状态。"""
    import src.pipeline.main as pipeline_main
    from src.pipeline.main import PowerPredictionPipeline

    tmpdir.mkdir(parents=True, exist_ok=True)
    pipeline = PowerPredictionPipeline()
    old_root = pipeline_main.PROJECT_ROOT
    old_graph_path = os.environ.get("MODELCOMBINE_GRAPH_STATE_PATH")
    isolated_graph = tmpdir / "graph_state.pkl"
    source_graph = PROJECT_ROOT / "reports" / "graph_state.pkl"
    if source_graph.exists():
        shutil.copy2(source_graph, isolated_graph)
    os.environ["MODELCOMBINE_GRAPH_STATE_PATH"] = str(isolated_graph)
    pipeline_main.PROJECT_ROOT = str(tmpdir)
    try:
        graph = pipeline.build_model_graph()
        t0 = time.perf_counter()
        selected, weights, scenario_id, path_id = pipeline.select_models_for_region(
            region, train, graph
        )
        selection_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred, performance = pipeline.fit_and_predict_region(
            region, selected, weights, train, test
        )
        training_seconds = time.perf_counter() - t1
    finally:
        pipeline_main.PROJECT_ROOT = old_root
        if old_graph_path is None:
            os.environ.pop("MODELCOMBINE_GRAPH_STATE_PATH", None)
        else:
            os.environ["MODELCOMBINE_GRAPH_STATE_PATH"] = old_graph_path
    if pred.empty:
        raise RuntimeError("legacy combinator produced no predictions")
    trained = [c.removeprefix("yhat_") for c in pred.columns if c.startswith("yhat_")]
    return {
        "models": list(selected),
        "trained_models": trained,
        "weights": dict(weights),
        "scenario_id": scenario_id,
        "path_id": path_id,
        "metrics": metric_summary(
            pred["load"].to_numpy(dtype=float), pred["yhat"].to_numpy(dtype=float)
        ),
        "selection_seconds": selection_seconds,
        "training_seconds": training_seconds,
        "elapsed_seconds": selection_seconds + training_seconds,
        "history_records_loaded": len(pipeline.historical_scenarios),
        "profiling": performance.get("_profiling") if isinstance(performance, dict) else None,
    }


def _run_once(rows: str, run_index: int, tmpdir: Path) -> Dict[str, Any]:
    from src.models.registry import model_registry
    from src.utils.io import load_yaml

    _seed_everything()
    train, test, region, split_meta = _load_and_split(rows)
    cfg = load_yaml(str(PROJECT_ROOT / "configs" / "pipeline.yaml"))
    candidates = [m for m in model_registry.get_available_models() if "blender" not in m.lower()]
    validation_days = _validation_days(train, rows)

    record: Dict[str, Any] = {
        "run_index": run_index,
        "status": "unknown",
        "input": {
            "region": region,
            "data_sha": _data_sha(pd.concat([train, test], ignore_index=True)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "split": split_meta,
            "validation_days": validation_days,
            "requested_candidate_models": candidates,
        },
    }
    try:
        t0 = time.perf_counter()
        bundle = build_region_prediction_bundle(
            region=region,
            train=train,
            test=test,
            candidate_models=candidates,
            validation_days=validation_days,
            model_params=cfg.get("models", {}),
        )
        record["prediction_bundle"] = {
            "elapsed_seconds": time.perf_counter() - t0,
            "surviving_models": list(bundle.model_cols),
            "failed_models": dict(bundle.metadata.get("failed_models") or {}),
            "n_val": int(len(bundle.df_val)),
            "n_test": int(len(bundle.df_test)),
        }

        record["combinator"] = _combinator_summary(train, test, region, tmpdir)
        record["protocol_a"] = _protocol_a_summary(bundle, region)
        record["protocol_b"] = _protocol_b_summary(
            bundle, region, diagnostic_name="current_full_registry_pool"
        )
        record["best_single"] = best_single_summaries(bundle)

        traditional_cols = traditional_model_cols(bundle.model_cols)
        if traditional_cols != bundle.model_cols:
            traditional_bundle = subset_bundle(bundle, traditional_cols)
            record["traditional_pool_diagnostic"] = _protocol_b_summary(
                traditional_bundle,
                region,
                diagnostic_name="same_predictions_without_deep_models",
            )
            record["traditional_pool_diagnostic"]["model_cols"] = traditional_cols
        else:
            record["traditional_pool_diagnostic"] = {
                "status": "not_run",
                "reason": "no deep model survived prediction bundle",
            }

        gate = quality_gate(
            protocol_b_mae=record["protocol_b"]["metrics"]["mae"],
            combinator_mae=record["combinator"]["metrics"]["mae"],
            validation_selected_single_mae=record["best_single"]["validation_selected"]["test"]["mae"],
        )
        record["quality_gate"] = gate

        removed = record["protocol_b"]["guard_evidence"].get("stability_removed_models") or {}
        unstable_late_present = any(
            "unstable_late" in str(reason).split(",") for reason in removed.values()
        )
        if not gate["passed"] and unstable_late_present:
            with _diagnostic_without_unstable_late():
                diagnostic = _protocol_b_summary(
                    bundle, region, diagnostic_name="unstable_late_disabled_in_memory"
                )
            diagnostic["product_behavior_changed"] = False
            record["unstable_late_diagnostic"] = diagnostic
        else:
            record["unstable_late_diagnostic"] = {
                "status": "not_run",
                "reason": (
                    "quality gate passed" if gate["passed"]
                    else "no candidate removed by unstable_late in this run"
                ),
            }
        record["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001 - 诊断报告需保留失败证据
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    return record


def _build_conclusions(runs: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    facts: List[str] = []
    inferences: List[str] = []
    unknowns: List[str] = []
    for run in runs:
        idx = run["run_index"]
        c_mae = run["combinator"]["metrics"]["mae"]
        b_mae = run["protocol_b"]["metrics"]["mae"]
        a_mae = run["protocol_a"]["metrics"]["mae"]
        facts.append(
            f"run{idx}: combinator MAE={c_mae:.6f}, Protocol A MAE={a_mae:.6f}, "
            f"Protocol B MAE={b_mae:.6f}, gate_passed={run['quality_gate']['passed']}"
        )
    if all(bool(run["quality_gate"]["passed"]) for run in runs):
        inferences.append("当前同轮证据支持 Protocol B 达到 combinator 与 validation 选单模型的 1% 容忍门槛。")
    else:
        inferences.append("至少一轮同轮证据确认 Protocol B 未达到质量切换门槛。")
    if len(runs) < 2:
        unknowns.append("只有一轮成功运行，尚不能判断重复运行稳定性。")
    if not any(
        isinstance(run.get("unstable_late_diagnostic"), dict)
        and run["unstable_late_diagnostic"].get("metrics")
        for run in runs
    ):
        unknowns.append("本轮未满足 unstable_late 因果诊断触发条件，不能归因 guard。")
    return {
        "observed_facts": facts,
        "evidence_supported_inferences": inferences,
        "still_unknown": unknowns,
    }


def worker_command(*, rows: str, run_index: int, output_path: Path) -> List[str]:
    """每轮用全新进程，避免深度模型/LightGBM 跨轮残留改变候选池。"""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--rows",
        str(rows),
        "--worker-run-index",
        str(run_index),
        "--worker-output",
        str(Path(output_path).resolve()),
    ]


def _run_worker(rows: str, run_index: int, output_path: Path) -> int:
    output = io.StringIO()
    with tempfile.TemporaryDirectory(prefix=f"ab_same_round_worker_{run_index}_") as tmp:
        tmpdir = Path(tmp)
        old_cwd = Path.cwd()
        os.chdir(tmpdir)
        try:
            with redirect_stdout(output), redirect_stderr(output):
                run = _run_once(rows, run_index, tmpdir / "artifacts")
        finally:
            os.chdir(old_cwd)
    run["log_tail"] = output.getvalue().splitlines()[-30:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="System A/B 同轮质量对照（Task 6A）")
    parser.add_argument("--rows", choices=("720", "full"), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--worker-run-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_run_index is not None or args.worker_output is not None:
        if args.worker_run_index is None or args.worker_output is None:
            parser.error("worker mode requires both --worker-run-index and --worker-output")
        return _run_worker(args.rows, args.worker_run_index, Path(args.worker_output))
    if not args.output:
        parser.error("--output is required outside worker mode")

    before = _snapshot_production_state()
    runs: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ab_same_round_parent_") as tmp:
        tmpdir = Path(tmp)
        for index in range(1, max(1, args.repeat) + 1):
            worker_output = tmpdir / f"run_{index}.json"
            completed = subprocess.run(
                worker_command(rows=args.rows, run_index=index, output_path=worker_output),
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"same-round worker {index} failed rc={completed.returncode}: "
                    f"{completed.stderr[-2000:]}"
                )
            runs.append(json.loads(worker_output.read_text(encoding="utf-8")))

    after = _snapshot_production_state()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "rows": args.rows,
        "repeat": args.repeat,
        "python": sys.executable,
        "random_seed": RANDOM_SEED,
        "runs": runs,
        "conclusions": _build_conclusions(runs),
        "guarded_state_before": before,
        "guarded_state_after": after,
        "readonly_guarantee_held": before == after,
    }
    validate_report(report)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[same-round] rows={args.rows} repeat={args.repeat} -> {output_path}")
    print(f"[same-round] readonly={report['readonly_guarantee_held']}")
    for run in runs:
        print(
            f"  run{run['run_index']}: combinator={run['combinator']['metrics']['mae']:.6f} "
            f"protocol_b={run['protocol_b']['metrics']['mae']:.6f} "
            f"gate={run['quality_gate']['passed']}"
        )
    return 0 if report["readonly_guarantee_held"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
