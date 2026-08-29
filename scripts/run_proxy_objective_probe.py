"""单任务代理口径探针（Task 8.3 Task 11）。

回答一个问题：stepwise 的 Ridge-only 代理与完整固定 pair 评估在第二个模型上的
排序分歧，来自哪个口径差异。

在同一个锚点（stepwise 第 0 步选中的模型）上，对每个候选第二模型用三种口径各算
一次组合误差：

- ``proxy_current``：现状代理——折外 blocked-CV MAE，alpha 取生产 stepwise 在
  **整个候选矩阵**上选出的单一值；
- ``proxy_subset_alpha``：同样是折外 blocked-CV MAE，但 alpha 按**该子集**重新选；
- ``full_pipeline``：完整固定 pair 评估（含 interaction 与 post-adjustment）的
  样本内 validation MAE，也就是候选诊断 best pair 所用的基准口径。

三者与基准排序对比，即可判断分歧是 alpha 口径、估计量口径，还是 interaction 本身。
test 标签只记录、不参与任何排序。不重训模型，只读传入的 baselines_v5 目录——
本脚本不做来源校验，调用方需自行保证 --pred-root 指向锁定的那份基线。
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = "task83-proxy-objective-probe.1"


def _rank(rows: List[Dict[str, Any]], key: str) -> List[str]:
    return [r["model"] for r in sorted(rows, key=lambda r: (r[key], r["model"]))]


def build_probe(
    *,
    pred_root: Path,
    raw_root: Optional[Path],
    dataset: str,
    horizon: int,
) -> Dict[str, Any]:
    from scripts.run_system_ab_shadow import _build_kg_model_candidates, build_task_matrix
    from src.eval.kg.config import KG_RIDGE_TEMPORAL_DECAY
    from src.eval.kg.data_io import _resolve_cv_config
    from src.eval.kg.drift import _compute_temporal_weights
    from src.eval.kg.model_selection import _fit_ridge_and_mae, evaluate_pair_on_validation
    from src.eval.kg.protocol_b import (
        evaluate_fixed_protocol_b_candidate,
        kg_combination_with_features,
    )
    from src.utils.blocked_cv import blocked_cv_select_alpha

    matrix = build_task_matrix(
        dataset=dataset, horizon=horizon,
        models=_build_kg_model_candidates(),
        pred_root=pred_root, raw_root=raw_root,
    )
    df_val = matrix["df_val_kg"]
    df_test = matrix["df_test_kg"]
    safe_models = list(matrix["safe_models"])

    production = kg_combination_with_features(
        df_val, df_test, matrix["df_raw_val"], matrix["df_raw_test"],
        safe_models, horizon,
        dataset_name=dataset, base_model_cols=matrix["base_model_cols"],
    )
    sel_meta = production["val"]["weight_meta"]["protocol_b_selection_meta"]
    stepwise_trace = sel_meta["stepwise_meta"]["trace"]
    anchor = stepwise_trace[0]["selected"]
    stepwise_alpha = float(sel_meta["stepwise_alpha"])

    y_val = df_val["y"].values
    sample_weight = _compute_temporal_weights(len(df_val), KG_RIDGE_TEMPORAL_DECAY)
    n_folds, min_train, gap = _resolve_cv_config(len(y_val), horizon)

    rows: List[Dict[str, Any]] = []
    for cand in sorted(m for m in safe_models if m != anchor):
        pair = [anchor, cand]
        X = df_val[pair].values

        proxy_current = _fit_ridge_and_mae(
            X, y_val, alpha=stepwise_alpha, horizon=horizon, sample_weight=sample_weight
        )

        subset_alpha, subset_cv = blocked_cv_select_alpha(
            X, y_val, alphas=None, n_folds=n_folds, min_train=min_train,
            positive=True, fit_intercept=False, sample_weight=sample_weight, gap=gap,
        )
        proxy_subset_alpha = _fit_ridge_and_mae(
            X, y_val, alpha=float(subset_alpha), horizon=horizon, sample_weight=sample_weight
        )

        in_sample = evaluate_pair_on_validation(
            df_val=df_val, pair=pair, horizon=horizon,
        )

        full = evaluate_fixed_protocol_b_candidate(
            df_val, df_test, matrix["df_raw_val"], matrix["df_raw_test"],
            selected_models=pair, horizon=horizon,
            dataset_name=dataset, base_model_cols=matrix["base_model_cols"],
        )

        rows.append({
            "model": cand,
            "pair": pair,
            "proxy_current": float(proxy_current),
            "proxy_current_alpha": stepwise_alpha,
            "proxy_subset_alpha": float(proxy_subset_alpha),
            "subset_alpha": float(subset_alpha),
            "ridge_in_sample": float(in_sample["validation_mae"]),
            "ridge_in_sample_eligible": bool(in_sample["eligible_pair"]),
            "full_pipeline": float(full["val"]["mae"]),
            "full_pipeline_eligible": bool(full["eligible_pair"]),
            # 只记录，不参与任何排序
            "test_mae_recorded_only": float(full["test"]["mae"]),
        })

    rankings = {
        key: _rank(rows, key)
        for key in ("proxy_current", "proxy_subset_alpha", "ridge_in_sample", "full_pipeline")
    }
    full_pipeline_subsets: List[Dict[str, Any]] = []
    for size in (2, 3):
        for models in combinations(sorted(safe_models), size):
            subset = kg_combination_with_features(
                df_val, df_test, matrix["df_raw_val"], matrix["df_raw_test"],
                list(models), horizon,
                dataset_name=dataset,
                base_model_cols=matrix["base_model_cols"],
                return_predictions=True,
                _fixed_selected_models=list(models),
                _skip_final_guard=True,
            )
            full_pipeline_subsets.append({
                "models": list(models),
                "validation_mae": float(subset["val"]["mae"]),
            })
    full_pipeline_subsets.sort(
        key=lambda row: (row["validation_mae"], tuple(row["models"]))
    )
    reference = rankings["full_pipeline"]
    agreement = {
        key: {
            "same_first_choice": order[0] == reference[0],
            "same_full_order": order == reference,
        }
        for key, order in rankings.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": {"dataset": dataset, "horizon": horizon},
        "safe_models": safe_models,
        "anchor": anchor,
        "stepwise_trace": stepwise_trace,
        "production_selected_models": list(production["val"]["selected_models"]),
        "production_protocol": production["protocol"],
        "candidates": rows,
        "rankings": rankings,
        "agreement_with_full_pipeline": agreement,
        "full_pipeline_subsets": full_pipeline_subsets,
        "notes": (
            "排序只用 validation；test_mae_recorded_only 仅记录。"
            "同一锚点下比较第二个模型的排序，用于定位代理与完整流水线的分歧来源。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--dataset", default="aemo_nsw")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_probe(
        pred_root=args.pred_root, raw_root=args.raw_root,
        dataset=args.dataset, horizon=args.horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"anchor = {report['anchor']}")
    for key, order in report["rankings"].items():
        mark = "  <= 基准" if key == "full_pipeline" else ""
        print(f"  {key:20s} {order}{mark}")
    for key, agree in report["agreement_with_full_pipeline"].items():
        if key == "full_pipeline":
            continue
        print(f"  {key:20s} 首选一致={agree['same_first_choice']} 全序一致={agree['same_full_order']}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
