#!/usr/bin/env python3
"""V1 相似度核心对照（SQLite 模型库 Piece 4）。

对每个真实 h=24 任务，比较三种关系选择方式在该任务真实 test 上的表现：

  Similarity Match  —— 用当前场景相似度检索到的历史组合。
  No Similarity     —— 忽略相似度，在兼容关系里按历史 validation MAE 最小选。
  Best Single       —— 该任务 validation 选出的最佳单模型。

只读运行：不训练、不改参数、不写数据库。运行一次即可，结果出来后不据此调参。

用法::

    python scripts/run_v1_similarity_comparison.py \
      --database modelcombine.sqlite3 \
      --pred-root baselines_v1 \
      --raw-root data/features \
      --datasets pjm aemo_vic aemo_nsw --horizons 24 \
      --output result/v1/similarity_comparison.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train_baselines as tb
from scripts.train_combinations_kg import (
    _build_kg_model_candidates,
    _forecast_origin_raw_frame,
    _scenario_signature,
)
from src.core.index import ScenarioIndex
from src.eval.combination_utils import load_predictions_safe
from src.models.artifacts import load_artifact
from src.storage.model_store import ModelStore

TASK_TYPE = "load_forecast"
BUSINESS_DOMAIN = "power_load"
TARGET = "load"


def _mae(pred: Any, truth: Any) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(truth)
    return float(np.mean(np.abs(pred[mask] - truth[mask])))


def _query_signature(raw_root: Path, dataset: str, horizon: int):
    """按调用方"根据预测发生前已有的历史负荷生成场景/数据特征"的口径生成签名。

    只读 validation 数据，绝不读取即将被评价的 test 真实负荷；否则 test 标签进入
    关系选择。构造方式与 _build_library_task 写入 scenario 签名时一致。
    """
    raw = pd.read_csv(raw_root / dataset / "val.csv")
    _x, y, ts, freq = tb.prepare_supervised(raw, TARGET, horizon)
    y_frame = pd.DataFrame({"timestamp": pd.to_datetime(ts).values, "y": np.asarray(y, dtype=float)})
    origin_frame = _forecast_origin_raw_frame(raw_root, dataset, "val", horizon)
    return _scenario_signature(y_frame, origin_frame, horizon, freq), freq


def _best_single(pred_root: Path, dataset: str, horizon: int, models: List[str]) -> Dict[str, Any]:
    df_val = load_predictions_safe(pred_root, dataset, horizon, models, "val")
    df_test = load_predictions_safe(pred_root, dataset, horizon, models, "test")
    common = [m for m in models if m in df_val.columns and m in df_test.columns]
    val_mae = {m: _mae(df_val[m], df_val["y"]) for m in common}
    best = min(val_mae, key=val_mae.get)
    return {"model": best, "validation_mae": val_mae[best], "test_mae": _mae(df_test[best], df_test["y"])}


def _apply_relation_to_task(
    store: ModelStore, relation: Dict[str, Any], query_dataset: str, horizon: int, raw_root: Path
) -> Dict[str, Any]:
    """把关系的组合器 + 其登记的模型，应用到 query 任务真实 test 上——事后评价，
    走的是与 run.py predict 完全一致的一份预测起点特征。"""
    combination = store.get_combination(relation["combination_id"])
    predictor = load_artifact(combination["artifact_path"])

    _y_x, y_true, ts, _freq = tb.prepare_supervised(
        pd.read_csv(raw_root / query_dataset / "test.csv"), TARGET, horizon
    )
    features = _forecast_origin_raw_frame(raw_root, query_dataset, "test", horizon)

    base_predictions: Dict[str, np.ndarray] = {}
    for member, member_type in zip(combination["members"], predictor.member_ids):
        model_row = store.get_model(member["model_id"])
        required = list(model_row["required_features"])
        missing = [column for column in required if column not in features.columns]
        if missing:
            return {
                "applicable": False,
                "reason": f"特征契约不兼容：缺少 {missing}",
                "relation_id": relation["relation_id"],
                "combination_id": relation["combination_id"],
                "scenario_id": relation["scenario_id"],
            }
        model = load_artifact(model_row["artifact_path"])
        base_predictions[member_type] = np.asarray(model.predict(features[required]), dtype=float)

    yhat = predictor.predict(base_predictions, features)
    return {
        "applicable": True,
        "relation_id": relation["relation_id"],
        "combination_id": relation["combination_id"],
        "scenario_id": relation["scenario_id"],
        "effective_members": [m["model_id"] for m in combination["members"]],
        "model_count": len(combination["members"]),
        "validation_mae": relation["validation_mae"],
        "recorded_test_mae": relation["test_mae"],
        "test_mae": _mae(yhat, y_true),
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def run(
    *,
    database: Path,
    pred_root: Path,
    raw_root: Path,
    datasets: List[str],
    horizons: List[int],
) -> Dict[str, Any]:
    store = ModelStore(str(database))
    models = list(dict.fromkeys(_build_kg_model_candidates() + ["seasonal_naive"]))
    report: Dict[str, Any] = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "commit": _git_commit(),
            "database": str(database),
            "pred_root": str(pred_root),
            "raw_root": str(raw_root),
        },
        "tasks": [],
    }
    try:
        for dataset in datasets:
            for horizon in horizons:
                signature, freq = _query_signature(raw_root, dataset, horizon)
                compatible = store.list_scenarios(
                    task_type=TASK_TYPE,
                    business_domain=BUSINESS_DOMAIN,
                    horizon=horizon,
                    freq=freq,
                )
                if not compatible:
                    raise RuntimeError(
                        f"no compatible scenario for {dataset} h={horizon} (freq={freq})"
                    )

                index = ScenarioIndex()
                for scenario in compatible:
                    index.add(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "signature": scenario["signature"],
                            "business_domain": scenario["business_domain"],
                        }
                    )
                ranked = index.query(signature=signature)
                similarity_scenario_id = ranked[0]["scenario_id"]
                similarity_score = float(ranked[0]["_score"])
                similarity_relation = store.list_relations_for_scenario(similarity_scenario_id)[0]

                all_relations: List[Dict[str, Any]] = []
                for scenario in compatible:
                    all_relations.extend(store.list_relations_for_scenario(scenario["scenario_id"]))
                no_similarity_relation = min(
                    all_relations, key=lambda r: (r["validation_mae"], r["relation_id"])
                )

                similarity = _apply_relation_to_task(
                    store, similarity_relation, dataset, horizon, raw_root
                )
                no_similarity = _apply_relation_to_task(
                    store, no_similarity_relation, dataset, horizon, raw_root
                )
                best_single = _best_single(pred_root, dataset, horizon, models)

                similarity["scenario_similarity_score"] = similarity_score
                for block in (similarity, no_similarity):
                    if block.get("applicable"):
                        block["ratio_vs_best_single"] = (
                            block["test_mae"] / best_single["test_mae"]
                            if best_single["test_mae"] > 0
                            else None
                        )

                report["tasks"].append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "query_signature": signature,
                        "similarity_match": similarity,
                        "no_similarity": no_similarity,
                        "best_single": best_single,
                        "selected_different": (
                            similarity_relation["relation_id"]
                            != no_similarity_relation["relation_id"]
                        ),
                    }
                )
    finally:
        store.close()

    similarity_ratios = [
        t["similarity_match"].get("ratio_vs_best_single")
        for t in report["tasks"]
        if t["similarity_match"].get("applicable")
    ]
    report["summary"] = {
        "n_tasks": len(report["tasks"]),
        "similarity_selected_different_relation": sum(
            1 for t in report["tasks"] if t["selected_different"]
        ),
        "similarity_better_than_best_single": sum(1 for r in similarity_ratios if r < 1.0 - 1e-9),
        "similarity_tie_best_single": sum(1 for r in similarity_ratios if abs(r - 1.0) <= 1e-9),
        "similarity_worse_than_best_single": sum(1 for r in similarity_ratios if r > 1.0 + 1e-9),
        "similarity_ratio_vs_best_single_by_task": {
            f"{t['dataset']}_h{t['horizon']}": t["similarity_match"].get("ratio_vs_best_single")
            for t in report["tasks"]
        },
        "no_similarity_applicable_by_task": {
            f"{t['dataset']}_h{t['horizon']}": t["no_similarity"].get("applicable")
            for t in report["tasks"]
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 相似度核心对照")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/features"))
    parser.add_argument("--datasets", nargs="*", default=["pjm", "aemo_vic", "aemo_nsw"])
    parser.add_argument("--horizons", nargs="*", type=int, default=[24])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run(
        database=args.database,
        pred_root=args.pred_root,
        raw_root=args.raw_root,
        datasets=args.datasets,
        horizons=args.horizons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\n完整报告: {args.output}")


if __name__ == "__main__":
    main()
