"""固定九任务候选诊断（Task 8.3 Task 6）。

严格复用 `baselines_v5`（不得重训），锁定来源校验后，对每个任务枚举单模型与
全部二模型组合（`evaluate_fixed_protocol_b_candidate` 绕过 selector/guard，只复用
拟合/interaction/post-adjustment），并按 **validation MAE only** 选出最佳二模型
作为固定枚举基准。生产 Protocol B 结果只作对照。test 标签仅记录、不参与选择。

报告补充关系反馈证据（Task 5）：每个最终模型的 oof_gain/polarity/skip_reason，
以及九任务正/负/中性事件数量与 guard 回退跳过的任务。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "task83-candidate-diagnostic.1"

# Global Constraints 锁定的来源哈希（固定 baselines_v5，不得重训）。
LOCKED_SEED = 42
LOCKED_PIPELINE_SHA256 = "200a7067f76d50c3356f79b8bce7873673d7aa0958d5a60b39818bfbd34f42ec"
LOCKED_PROVENANCE_SHA256 = "5d850b7c6782472bc55c2a9c877cb5291c88cb3da02b9f33448eb234c24a9df3"
LOCKED_DATA_SHA256 = {
    "pjm": "53043c2e36fbf053dbdc8e9081583d22bf1ac4461773c0adba131aaed1866227",
    "aemo_vic": "1f5b5c965936dc1433265b88401d373a059e1c5457d5d2e0ff8f10b7a8385ad2",
    "aemo_nsw": "36b498f96472b7a95da365dab2577aae7989523c046516ee3f7eecd229fe1b85",
}
EXPECTED_ARTIFACT_ENTRIES = 72
EXPECTED_FILE_HASHES = 225


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sha256(path: Path, expected: str, *, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def select_validation_best_pair(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """按 validation MAE 选最佳**有资格**二模型；同分用有序模型元组作次级键。"""
    eligible = [r for r in rows if r.get("eligible_pair") is True]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (float(r["validation_mae"]), tuple(r["models"])))


def summarize_diagnostic_coverage(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """统计 validation 最佳 pair 与 Protocol B 选择的重合，以及显式 conflict 消费。"""
    same_tasks: List[str] = []
    conflict_tasks = 0
    for task in tasks:
        bp = task.get("best_pair") or {}
        pb = task.get("protocol_b") or {}
        bp_models = set(bp.get("models") or [])
        pb_models = set(pb.get("models") or [])
        if bp_models and bp_models == pb_models:
            same_tasks.append(task["task_id"])
        if int(task.get("explicit_conflict_edges_consumed") or 0) > 0:
            conflict_tasks += 1
    return {
        "best_pair_same_as_protocol_b_count": len(same_tasks),
        "best_pair_same_as_protocol_b_tasks": same_tasks,
        "tasks_with_explicit_conflict_edges": conflict_tasks,
    }


def _hash_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def verify_locked_sources(
    *,
    pred_root: Path,
    pipeline_config: Path,
    feature_root: Path,
) -> Dict[str, Any]:
    """校验锁定来源：provenance 与数据哈希、pipeline 哈希、artifact 数量与逐文件哈希。"""
    from scripts.train_baselines import load_verified_baseline_provenance

    provenance = load_verified_baseline_provenance(pred_root, pipeline_config)
    verify_sha256(
        pred_root / "baseline_provenance.json",
        LOCKED_PROVENANCE_SHA256,
        label="baseline_provenance.json",
    )
    verify_sha256(pipeline_config, LOCKED_PIPELINE_SHA256, label="pipeline config")
    data_hashes: Dict[str, str] = {}
    for ds, expected in LOCKED_DATA_SHA256.items():
        data_hashes[ds] = verify_sha256(
            feature_root / ds / "train.csv", expected, label=f"{ds} train.csv"
        )

    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("baseline provenance artifacts must be a list")
    if len(artifacts) != EXPECTED_ARTIFACT_ENTRIES:
        raise ValueError(
            f"expected {EXPECTED_ARTIFACT_ENTRIES} artifact entries, got {len(artifacts)}"
        )
    file_hash_count = 0
    for record in artifacts:
        files = record.get("files") if isinstance(record, dict) else {}
        if not isinstance(files, dict):
            continue
        dataset = record.get("dataset")
        for file_name, expected_hash in files.items():
            path = pred_root / str(dataset) / str(file_name)
            verify_sha256(path, str(expected_hash), label=f"artifact {dataset}/{file_name}")
            file_hash_count += 1
    if file_hash_count != EXPECTED_FILE_HASHES:
        raise ValueError(
            f"expected {EXPECTED_FILE_HASHES} file hashes, got {file_hash_count}"
        )
    return {
        "status": "verified",
        "provenance_sha256": LOCKED_PROVENANCE_SHA256,
        "pipeline_sha256": LOCKED_PIPELINE_SHA256,
        "data_sha256": data_hashes,
        "artifact_entries": len(artifacts),
        "file_hashes": file_hash_count,
        "seed": LOCKED_SEED,
    }


def _record_single(df_val: pd.DataFrame, df_test: pd.DataFrame, model: str) -> Dict[str, Any]:
    y_val = df_val["y"].to_numpy()
    y_test = df_test["y"].to_numpy()
    return {
        "model": model,
        "validation_mae": float(np.mean(np.abs(df_val[model].to_numpy() - y_val))),
        "test_mae": float(np.mean(np.abs(df_test[model].to_numpy() - y_test))),
    }


def _relation_feedback_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    fb = raw.get("relation_feedback")
    if not isinstance(fb, dict):
        fb = {}
    by_model = fb.get("by_model") if isinstance(fb.get("by_model"), dict) else {}
    per_model = []
    for model, item in by_model.items():
        if not isinstance(item, dict):
            continue
        per_model.append({
            "model": model,
            "oof_gain": item.get("oof_gain"),
            "validation_gain": item.get("validation_gain"),
            "final_weight": item.get("final_weight"),
            "polarity": item.get("polarity"),
            "magnitude": item.get("magnitude"),
            "skip_reason": item.get("skip_reason"),
        })
    return {
        "eligible": bool(fb.get("eligible")),
        "skip_reason": fb.get("skip_reason"),
        "evidence_mode": fb.get("evidence_mode"),
        "per_model": per_model,
    }


def build_diagnostic_report(
    *,
    pred_root: Path,
    raw_root: Optional[Path],
    feature_root: Path,
    pipeline_config: Path,
    tasks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """对 9 任务枚举单模型/全部二模型，并输出候选诊断报告。"""
    from scripts.run_system_ab_shadow import _build_kg_model_candidates, build_task_matrix
    from src.eval.kg.protocol_b import (
        evaluate_fixed_protocol_b_candidate,
        kg_combination_with_features,
    )

    locked_sources = verify_locked_sources(
        pred_root=pred_root, pipeline_config=pipeline_config, feature_root=feature_root
    )
    models = _build_kg_model_candidates()
    records: List[Dict[str, Any]] = []
    for spec in tasks:
        dataset, horizon = spec["dataset"], spec["horizon"]
        matrix = build_task_matrix(
            dataset=dataset,
            horizon=horizon,
            models=models,
            pred_root=pred_root,
            raw_root=raw_root,
        )
        safe_models = list(matrix["safe_models"])
        singles = [
            _record_single(matrix["df_val_kg"], matrix["df_test_kg"], m)
            for m in safe_models
        ]
        pairs: List[Dict[str, Any]] = []
        for left, right in itertools.combinations(sorted(safe_models), 2):
            raw = evaluate_fixed_protocol_b_candidate(
                matrix["df_val_kg"], matrix["df_test_kg"],
                matrix["df_raw_val"], matrix["df_raw_test"],
                selected_models=[left, right], horizon=horizon,
                dataset_name=dataset, base_model_cols=matrix["base_model_cols"],
            )
            pairs.append({
                "models": [left, right],
                "validation_mae": float((raw.get("val") or {}).get("mae", float("inf"))),
                "test_mae": float((raw.get("test") or {}).get("mae", float("inf"))),
                "eligible_pair": bool(raw.get("eligible_pair")),
                "degenerate_reason": raw.get("degenerate_reason"),
                "guard_would_fallback_to": raw.get("guard_would_fallback_to"),
                "guard_would_fallback_reason": raw.get("guard_would_fallback_reason"),
            })

        production_raw = kg_combination_with_features(
            matrix["df_val_kg"], matrix["df_test_kg"],
            matrix["df_raw_val"], matrix["df_raw_test"],
            safe_models, horizon,
            dataset_name=dataset, base_model_cols=matrix["base_model_cols"],
        )
        prod_split = production_raw.get("val") or production_raw.get("test") or {}
        prod_models = list(prod_split.get("selected_models") or [])
        prod_weight_meta = prod_split.get("weight_meta") or {}
        selection_meta = (prod_weight_meta.get("protocol_b_selection_meta") or {})
        explicit_conflict = int(selection_meta.get("explicit_conflict_edges_consumed") or 0)
        guard_config = prod_weight_meta.get("guard_config") or {}
        prod_guard_target = (
            guard_config.get("final_fallback_target")
            if isinstance(guard_config, dict)
            else None
        )
        prod_guard_reason = (
            guard_config.get("final_fallback_reason")
            if isinstance(guard_config, dict)
            else None
        )

        best_single = min(singles, key=lambda r: (r["validation_mae"], r["model"]))
        best_pair = select_validation_best_pair(pairs)
        best_pair_record = None
        if best_pair is not None:
            best_pair_record = {
                "models": list(best_pair["models"]),
                "validation_mae": float(best_pair["validation_mae"]),
                "selection_uses_test_labels": False,
                "selection_source": "validation_mae_only",
                "test": {"mae": float(best_pair["test_mae"])},
            }
        bp_models = set(best_pair_record["models"]) if best_pair_record else set()

        records.append({
            "task_id": f"{dataset}_h{horizon}",
            "dataset": dataset,
            "horizon": int(horizon),
            "matrix_hashes": {
                "df_val_kg": _hash_frame(matrix["df_val_kg"]),
                "df_test_kg": _hash_frame(matrix["df_test_kg"]),
            },
            "safe_models": safe_models,
            "singles": singles,
            "pairs": pairs,
            "best_single": best_single,
            "best_pair": best_pair_record,
            "protocol_b": {
                "models": prod_models,
                "relation_feedback": _relation_feedback_summary(production_raw),
                "guard_fallback_target": prod_guard_target,
                "guard_fallback_reason": prod_guard_reason,
            },
            "best_pair_same_as_protocol_b": bool(bp_models and bp_models == set(prod_models)),
            "explicit_conflict_edges_consumed": explicit_conflict,
        })

    coverage = summarize_diagnostic_coverage(records)
    # 关系反馈汇总（Task 5 补充）
    relation_event_counts = {"positive": 0, "negative": 0, "neutral": 0}
    guard_fallback_skipped_tasks: List[str] = []
    for rec in records:
        fb = rec["protocol_b"]["relation_feedback"]
        if fb.get("skip_reason") and str(fb["skip_reason"]).startswith("guard_fallback:"):
            guard_fallback_skipped_tasks.append(rec["task_id"])
        for item in fb.get("per_model") or []:
            pol = item.get("polarity")
            if pol in relation_event_counts:
                relation_event_counts[pol] += 1
    coverage.update({
        "relation_event_counts": relation_event_counts,
        "guard_fallback_skipped_tasks": guard_fallback_skipped_tasks,
        "explicit_conflict_mechanism_status": (
            "real_v6_triggered"
            if coverage["tasks_with_explicit_conflict_edges"] > 0
            else "unit_and_wiring_only"
        ),
    })

    report = {
        "schema_version": SCHEMA_VERSION,
        "locked_sources": locked_sources,
        "tasks": records,
        "summary": coverage,
    }
    validate_diagnostic_schema(report)
    return report


def validate_diagnostic_schema(report: Dict[str, Any]) -> None:
    """要求九条记录包含固定字段；最佳 pair 选择不得读 test 标签。"""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    tasks = report.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 9:
        raise ValueError("expected exactly 9 task records")
    required = (
        "task_id", "dataset", "horizon", "matrix_hashes", "safe_models",
        "singles", "pairs", "best_single", "best_pair", "protocol_b",
        "best_pair_same_as_protocol_b", "explicit_conflict_edges_consumed",
    )
    for task in tasks:
        for key in required:
            if key not in task:
                raise ValueError(f"task {task.get('task_id')} missing key {key}")
        bp = task["best_pair"]
        if bp is not None:
            if bp.get("selection_uses_test_labels") is not False:
                raise ValueError("best_pair must not use test labels")
            if bp.get("selection_source") != "validation_mae_only":
                raise ValueError("best_pair selection_source must be validation_mae_only")
            if "test" not in bp:
                raise ValueError("best_pair must record test only after selection")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    from scripts.run_system_ab_shadow import build_task_specs

    specs = build_task_specs()
    if len(specs) != 9:
        print(f"[warn] 期望 9 任务，实际 {len(specs)}", file=sys.stderr)

    report = build_diagnostic_report(
        pred_root=args.pred_root,
        raw_root=args.raw_root,
        feature_root=args.feature_root,
        pipeline_config=args.pipeline_config,
        tasks=specs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        "best_pair_same_as_protocol_b: "
        f"{summary['best_pair_same_as_protocol_b_count']}/9 "
        f"{summary['best_pair_same_as_protocol_b_tasks']}"
    )
    print(f"tasks_with_explicit_conflict_edges: {summary['tasks_with_explicit_conflict_edges']}")
    print(f"relation_event_counts: {summary['relation_event_counts']}")
    print(f"guard_fallback_skipped_tasks: {summary['guard_fallback_skipped_tasks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
