#!/usr/bin/env python3
"""
Supplementary runtime benchmark for formal ModelCombine runs.

Measures task-level runtime on the 9-task matrix:
  - static_weight_safe
  - stacking_safe
  - rl_qms
  - mole_router
  - iTransformer (checkpoint inference replay)
  - kg_protocol_a / kg_protocol_b (from kg_results.json runtime fields)

Outputs:
  <run_root>/reports/analysis/runtime_supplement_strict.json
  <run_root>/reports/analysis/runtime_supplement_strict.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.combination_utils import fit_ridge_robust
from src.utils.blocked_cv import blocked_cv_select_alpha
from strategies.mole_router import MoLERouterStrategy
from strategies.rl_qms import RLQMSStrategy


DATASETS = ["pjm", "aemo_vic", "aemo_nsw"]
HORIZONS = [1, 6, 24]
EXCLUDE_COLS = {"row_id", "timestamp", "y"}


def _fmt_ms(x: float | None, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{float(x):.{digits}f}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _timing_stats(values_ms: List[float]) -> Dict[str, Any]:
    arr = [float(v) for v in values_ms if np.isfinite(v)]
    if not arr:
        return {
            "runs_ms": [],
            "mean_ms": None,
            "max_ms": None,
            "min_ms": None,
            "std_ms": None,
        }
    return {
        "runs_ms": arr,
        "mean_ms": float(statistics.fmean(arr)),
        "max_ms": float(max(arr)),
        "min_ms": float(min(arr)),
        "std_ms": float(statistics.pstdev(arr)) if len(arr) > 1 else 0.0,
    }


def _measure(fn, repeats: int = 5, warmup: int = 1) -> Dict[str, Any]:
    for _ in range(max(0, warmup)):
        fn()
    runs = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1000.0)
    return _timing_stats(runs)


def _task_iter() -> List[Tuple[str, int]]:
    out = []
    for ds in DATASETS:
        for h in HORIZONS:
            out.append((ds, h))
    return out


def _load_task_frames(run_root: Path, dataset: str, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    ds_root = run_root / "reports" / "modelcombine" / dataset
    val_path = ds_root / f"val_base_h{horizon}.csv"
    test_path = ds_root / f"test_base_h{horizon}.csv"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing base files: {val_path}, {test_path}")

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    model_cols = [c for c in val_df.columns if c not in EXCLUDE_COLS]
    if not model_cols:
        raise ValueError(f"No model columns for {dataset} h={horizon}")

    # Fill NaNs by val means to keep protocol consistent with eval stage.
    col_means = val_df[model_cols].mean(numeric_only=True)
    val_df[model_cols] = val_df[model_cols].fillna(col_means)
    test_df[model_cols] = test_df[model_cols].fillna(col_means)
    return val_df, test_df, model_cols


def _fit_static_weight_safe(P_val: np.ndarray, y_val: np.ndarray):
    alpha, _ = blocked_cv_select_alpha(
        P_val,
        y_val,
        n_folds=3,
        min_train=50,
        positive=True,
        fit_intercept=False,
        sample_weight=None,
    )
    reg, solver_meta = fit_ridge_robust(
        P_val,
        y_val,
        alpha=alpha,
        positive=True,
        fit_intercept=False,
        sample_weight=None,
    )
    return reg, alpha, solver_meta


def _fit_stacking_safe(P_val: np.ndarray, y_val: np.ndarray):
    alpha, _ = blocked_cv_select_alpha(
        P_val,
        y_val,
        n_folds=3,
        min_train=50,
        positive=True,
        fit_intercept=True,
        sample_weight=None,
    )
    reg, solver_meta = fit_ridge_robust(
        P_val,
        y_val,
        alpha=alpha,
        positive=True,
        fit_intercept=True,
        sample_weight=None,
    )
    return reg, alpha, solver_meta


def _extract_itrain_train_cmds(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing iTransformer log: {log_path}")
    cmds: List[Dict[str, Any]] = []
    model_pat = re.compile(r"^mcit_(.+)_h(\d+)_\d+$")

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("[CMD] "):
                continue
            raw = line[len("[CMD] ") :]
            toks = shlex.split(raw)
            if "--is_training" not in toks:
                continue
            idx = toks.index("--is_training")
            if idx + 1 >= len(toks) or toks[idx + 1] != "1":
                continue
            if "--model_id" not in toks:
                continue
            midx = toks.index("--model_id")
            if midx + 1 >= len(toks):
                continue
            model_id = toks[midx + 1]
            m = model_pat.match(model_id)
            if not m:
                continue
            ds = m.group(1)
            h = int(m.group(2))
            cmds.append({"dataset": ds, "horizon": h, "tokens": toks})
    if not cmds:
        raise RuntimeError("No iTransformer train commands found in log")
    return cmds


def _convert_to_eval_test_cmd(tokens: List[str]) -> List[str]:
    out = list(tokens)
    if "--is_training" in out:
        idx = out.index("--is_training")
        if idx + 1 < len(out):
            out[idx + 1] = "0"
        else:
            out.append("0")
    else:
        out.extend(["--is_training", "0"])

    if "--eval_split" in out:
        idx = out.index("--eval_split")
        if idx + 1 < len(out):
            out[idx + 1] = "test"
        else:
            out.append("test")
    else:
        out.extend(["--eval_split", "test"])
    return out


def _aggregate_method(per_task: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    rows = []
    for item in per_task:
        if key in item and isinstance(item[key], dict):
            rows.append(item[key])
    vals_total = [r.get("total_inference_ms") for r in rows if r.get("total_inference_ms") is not None]
    vals_route = [r.get("routing_ms") for r in rows if r.get("routing_ms") is not None]
    return {
        "tasks": len(rows),
        "total_avg_ms": float(statistics.fmean(vals_total)) if vals_total else None,
        "total_max_ms": float(max(vals_total)) if vals_total else None,
        "total_min_ms": float(min(vals_total)) if vals_total else None,
        "routing_avg_ms": float(statistics.fmean(vals_route)) if vals_route else None,
        "routing_max_ms": float(max(vals_route)) if vals_route else None,
        "routing_min_ms": float(min(vals_route)) if vals_route else None,
    }


def _bench_methods(run_root: Path, repeats: int, warmup: int, mole_epochs: int, mole_batch_size: int) -> List[Dict[str, Any]]:
    per_task: List[Dict[str, Any]] = []
    for ds, h in _task_iter():
        val_df, test_df, model_cols = _load_task_frames(run_root, ds, h)
        P_val = val_df[model_cols].to_numpy(dtype=float)
        P_test = test_df[model_cols].to_numpy(dtype=float)
        y_val = val_df["y"].to_numpy(dtype=float)
        y_test = test_df["y"].to_numpy(dtype=float)

        task = {"dataset": ds, "horizon": int(h)}

        print(f"[task] benchmarking {ds} h={h}", flush=True)

        # static_weight_safe
        t0 = time.perf_counter()
        reg_sw, alpha_sw, _sw_solver = _fit_static_weight_safe(P_val, y_val)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        pred_stats = _measure(lambda: reg_sw.predict(P_test), repeats=repeats, warmup=warmup)
        task["static_weight_safe"] = {
            "fit_ms": float(fit_ms),
            "routing_ms": 0.0,
            "total_inference_ms": pred_stats["mean_ms"],
            "predict_detail": pred_stats,
            "alpha": float(alpha_sw),
            "model_count": len(model_cols),
        }

        # stacking_safe
        t0 = time.perf_counter()
        reg_st, alpha_st, _st_solver = _fit_stacking_safe(P_val, y_val)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        pred_stats = _measure(lambda: reg_st.predict(P_test), repeats=repeats, warmup=warmup)
        task["stacking_safe"] = {
            "fit_ms": float(fit_ms),
            "routing_ms": 0.0,
            "total_inference_ms": pred_stats["mean_ms"],
            "predict_detail": pred_stats,
            "alpha": float(alpha_st),
            "model_count": len(model_cols),
        }

        # rl_qms
        rl = RLQMSStrategy(
            Nq=72,
            Nsp=4,
            alpha=0.1,
            gamma=0.8,
            Ne=100,
            em_metric="ape",
            seed=42,
            warm_start_from_val=True,
            active_models=model_cols,
            switch_penalty=0.5,
            test_epsilon=0.0,
        )
        t0 = time.perf_counter()
        rl.fit(P_val, y_val, model_names=model_cols)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        pred_stats = _measure(
            lambda: rl.predict(P_test, y_test=y_test, model_names=model_cols),
            repeats=repeats,
            warmup=warmup,
        )
        task["rl_qms"] = {
            "fit_ms": float(fit_ms),
            "routing_ms": pred_stats["mean_ms"],
            "total_inference_ms": pred_stats["mean_ms"],
            "predict_detail": pred_stats,
            "model_count": len(model_cols),
            "note": "routing measured as online policy prediction call",
        }

        # mole_router
        mole = MoLERouterStrategy(
            hidden_dim=16,
            epochs=int(mole_epochs),
            lr=1e-2,
            batch_size=int(mole_batch_size),
            head_dropout=0.1,
            seed=42,
            active_models=model_cols,
            temperature=1.5,
            temporal_holdout=0.2,
            weight_clip=0.8,
        )
        t0 = time.perf_counter()
        mole.fit(P_val, y_val, ctx_val=val_df, model_names=model_cols)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        pred_stats = _measure(
            lambda: mole.predict(P_test, y_test=y_test, ctx_test=test_df, model_names=model_cols),
            repeats=repeats,
            warmup=warmup,
        )
        task["mole_router"] = {
            "fit_ms": float(fit_ms),
            "routing_ms": pred_stats["mean_ms"],
            "total_inference_ms": pred_stats["mean_ms"],
            "predict_detail": pred_stats,
            "model_count": len(model_cols),
            "note": "routing measured as router forward + weighted fusion",
        }

        per_task.append(task)
        print(
            f"[runtime] {ds} h={h} "
            f"SW={_fmt_ms(task['static_weight_safe']['total_inference_ms'])}ms "
            f"ST={_fmt_ms(task['stacking_safe']['total_inference_ms'])}ms "
            f"RL={_fmt_ms(task['rl_qms']['total_inference_ms'])}ms "
            f"MoLE={_fmt_ms(task['mole_router']['total_inference_ms'])}ms"
        , flush=True)
    return per_task


def _load_kg_runtime(run_root: Path) -> List[Dict[str, Any]]:
    kg_path = run_root / "reports" / "combos_kg" / "kg_results.json"
    if not kg_path.exists():
        raise FileNotFoundError(f"Missing kg_results.json: {kg_path}")
    with kg_path.open("r", encoding="utf-8") as f:
        kg = json.load(f)

    out = []
    for ds in DATASETS:
        ds_obj = kg.get(ds, {})
        for h in HORIZONS:
            hobj = ds_obj.get(str(h), {}) if isinstance(ds_obj, dict) else {}
            meta = hobj.get("_meta", {}) if isinstance(hobj, dict) else {}
            a_sec = meta.get("runtime_protocol_a_sec")
            b_sec = meta.get("runtime_protocol_b_sec")
            out.append(
                {
                    "dataset": ds,
                    "horizon": int(h),
                    "kg_protocol_a": float(a_sec) * 1000.0 if a_sec is not None else None,
                    "kg_protocol_b": float(b_sec) * 1000.0 if b_sec is not None else None,
                }
            )
    return out


def _bench_itransformer(
    run_root: Path,
    itrans_root: Path,
    dry_run: bool = False,
    force_cpu: bool = True,
) -> List[Dict[str, Any]]:
    log_path = run_root / "logs" / "04a_itransformer.log"
    cmds = _extract_itrain_train_cmds(log_path)
    cmds = sorted(cmds, key=lambda x: (x["dataset"], x["horizon"]))
    run_log = run_root / "reports" / "analysis" / "runtime_supplement_itransformer_replay.log"
    _ensure_dir(run_log.parent)

    rows = []
    with run_log.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== iTransformer replay start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for item in cmds:
            ds = item["dataset"]
            h = int(item["horizon"])
            cmd = _convert_to_eval_test_cmd(item["tokens"])
            lf.write(f"\n[task] {ds} h={h}\n")
            lf.write("[cmd] " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            lf.flush()

            if dry_run:
                elapsed_ms = None
            else:
                env = os.environ.copy()
                if force_cpu:
                    # Disable CUDA visibility to replay inference on CPU when GPU is busy.
                    env["CUDA_VISIBLE_DEVICES"] = ""
                t0 = time.perf_counter()
                subprocess.run(cmd, cwd=str(itrans_root), check=True, stdout=lf, stderr=lf, env=env)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            rows.append(
                {
                    "dataset": ds,
                    "horizon": h,
                    "routing_ms": None,
                    "total_inference_ms": float(elapsed_ms) if elapsed_ms is not None else None,
                    "note": "checkpoint replay with run.py --is_training 0 --eval_split test",
                }
            )
            print(f"[runtime] iTransformer {ds} h={h} -> {_fmt_ms(elapsed_ms)} ms", flush=True)
        lf.write(f"\n=== iTransformer replay end {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return rows


def _build_markdown(payload: Dict[str, Any]) -> str:
    s = payload["summary"]
    protocol = payload["protocol"]
    lines: List[str] = []
    lines.append("# Runtime Supplement (Strict)")
    lines.append("")
    lines.append(f"- run_root: `{payload['run_root']}`")
    lines.append(f"- benchmark_time: `{payload['generated_at']}`")
    lines.append(f"- protocol: {protocol}")
    lines.append("")
    lines.append("| Method | Device | Routing Time (ms) | Total Inference Time (ms) | Coverage |")
    lines.append("|---|---|---:|---:|---:|")

    def row(method: str, device: str, routing: str, total: str, coverage: int) -> None:
        lines.append(f"| {method} | {device} | {routing} | {total} | {coverage}/9 |")

    sw = s["static_weight_safe"]
    st = s["stacking_safe"]
    rl = s["rl_qms"]
    mo = s["mole_router"]
    it = s["itransformer"]
    kg = s["kg_protocol"]

    row(
        "Static Weight Safe",
        "CPU",
        "0 (fixed rule)",
        f"{_fmt_ms(sw['total_avg_ms'])} avg / {_fmt_ms(sw['total_max_ms'])} max",
        sw["tasks"],
    )
    row(
        "Stacking Safe",
        "CPU",
        "0 (fixed linear combiner)",
        f"{_fmt_ms(st['total_avg_ms'])} avg / {_fmt_ms(st['total_max_ms'])} max",
        st["tasks"],
    )
    row(
        "RL-QMS",
        "CPU",
        f"{_fmt_ms(rl['routing_avg_ms'])} avg / {_fmt_ms(rl['routing_max_ms'])} max",
        f"{_fmt_ms(rl['total_avg_ms'])} avg / {_fmt_ms(rl['total_max_ms'])} max",
        rl["tasks"],
    )
    row(
        "MoLE Router",
        "CPU",
        f"{_fmt_ms(mo['routing_avg_ms'])} avg / {_fmt_ms(mo['routing_max_ms'])} max",
        f"{_fmt_ms(mo['total_avg_ms'])} avg / {_fmt_ms(mo['total_max_ms'])} max",
        mo["tasks"],
    )
    row(
        "iTransformer",
        "GPU",
        "N/A (single-model)",
        f"{_fmt_ms(it['total_avg_ms'])} avg / {_fmt_ms(it['total_max_ms'])} max",
        it["tasks"],
    )
    row(
        "ModelCombine Protocol A",
        "CPU",
        "Included in total (no sub-timer)",
        f"{_fmt_ms(kg['protocol_a_avg_ms'])} avg / {_fmt_ms(kg['protocol_a_max_ms'])} max",
        kg["tasks"],
    )
    row(
        "ModelCombine Protocol B",
        "CPU",
        "Included in total (no sub-timer)",
        f"{_fmt_ms(kg['protocol_b_avg_ms'])} avg / {_fmt_ms(kg['protocol_b_max_ms'])} max",
        kg["tasks"],
    )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Static/RL-QMS/MoLE timing uses the formal run's `val_base_h*` and `test_base_h*` artifacts.")
    lines.append("- `total_inference_ms` is measured on test prediction calls; fit times are reported in JSON for audit.")
    lines.append("- iTransformer timing is replayed by rerunning `run.py --is_training 0 --eval_split test` from logged training commands.")
    lines.append("- KG Protocol A/B timing is read from `kg_results.json` fields `runtime_protocol_a_sec` / `runtime_protocol_b_sec`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True, help="formal run root, e.g. result/0307/<run>")
    parser.add_argument("--itrans-root", type=Path, default=PROJECT_ROOT / "Comparison_Algorithm" / "iTransformer-main")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-itransformer", action="store_true")
    parser.add_argument("--itrans-dry-run", action="store_true")
    parser.add_argument("--mole-epochs", type=int, default=40, help="MoLE fit epochs for timing prep")
    parser.add_argument("--mole-batch-size", type=int, default=512, help="MoLE fit batch size for timing prep")
    parser.add_argument("--itrans-force-cpu", action="store_true", default=True, help="Force iTransformer replay on CPU")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"run_root not found: {run_root}")
    out_dir = run_root / "reports" / "analysis"
    _ensure_dir(out_dir)

    per_task = _bench_methods(
        run_root,
        repeats=args.repeats,
        warmup=args.warmup,
        mole_epochs=args.mole_epochs,
        mole_batch_size=args.mole_batch_size,
    )
    kg_rows = _load_kg_runtime(run_root)

    itr_rows: List[Dict[str, Any]] = []
    if not args.skip_itransformer:
        itr_rows = _bench_itransformer(
            run_root,
            args.itrans_root.resolve(),
            dry_run=args.itrans_dry_run,
            force_cpu=args.itrans_force_cpu,
        )

    # Merge KG + iTransformer into per-task records.
    task_index = {(x["dataset"], int(x["horizon"])): x for x in per_task}
    for row in kg_rows:
        k = (row["dataset"], int(row["horizon"]))
        if k in task_index:
            task_index[k]["kg_protocol_a"] = {
                "routing_ms": None,
                "total_inference_ms": row["kg_protocol_a"],
            }
            task_index[k]["kg_protocol_b"] = {
                "routing_ms": None,
                "total_inference_ms": row["kg_protocol_b"],
            }
    for row in itr_rows:
        k = (row["dataset"], int(row["horizon"]))
        if k in task_index:
            task_index[k]["itransformer"] = {
                "routing_ms": row.get("routing_ms"),
                "total_inference_ms": row.get("total_inference_ms"),
                "note": row.get("note"),
            }

    merged_tasks = [task_index[(ds, h)] for ds, h in _task_iter()]

    method_summary = {
        "static_weight_safe": _aggregate_method(merged_tasks, "static_weight_safe"),
        "stacking_safe": _aggregate_method(merged_tasks, "stacking_safe"),
        "rl_qms": _aggregate_method(merged_tasks, "rl_qms"),
        "mole_router": _aggregate_method(merged_tasks, "mole_router"),
        "itransformer": _aggregate_method(merged_tasks, "itransformer"),
    }

    kg_a = [
        t.get("kg_protocol_a", {}).get("total_inference_ms")
        for t in merged_tasks
        if t.get("kg_protocol_a", {}).get("total_inference_ms") is not None
    ]
    kg_b = [
        t.get("kg_protocol_b", {}).get("total_inference_ms")
        for t in merged_tasks
        if t.get("kg_protocol_b", {}).get("total_inference_ms") is not None
    ]
    method_summary["kg_protocol"] = {
        "tasks": int(min(len(kg_a), len(kg_b))),
        "protocol_a_avg_ms": float(statistics.fmean(kg_a)) if kg_a else None,
        "protocol_a_max_ms": float(max(kg_a)) if kg_a else None,
        "protocol_b_avg_ms": float(statistics.fmean(kg_b)) if kg_b else None,
        "protocol_b_max_ms": float(max(kg_b)) if kg_b else None,
    }

    payload: Dict[str, Any] = {
        "run_root": str(run_root),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": (
            f"strict 9-task benchmark; repeats={args.repeats}, warmup={args.warmup}; "
            "external methods timed on existing base predictions; iTransformer timed via checkpoint replay"
        ),
        "summary": method_summary,
        "tasks": merged_tasks,
    }

    out_json = out_dir / "runtime_supplement_strict.json"
    out_md = out_dir / "runtime_supplement_strict.md"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    out_md.write_text(_build_markdown(payload), encoding="utf-8")

    print(f"[saved] {out_json}")
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
