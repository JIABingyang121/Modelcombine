#!/usr/bin/env python3
"""Stage 2 归因：把"相似度选错"和"三条历史组合本身都不好"分开（只读）。

对每个查询窗口 Q，分别算出它与 S1/S2/S3 三个历史场景的相似度，以及**强制**使用
每一条历史关系时的反事实 MAE：

```text
T1 × {S1, S2, S3} -> (相似度, 反事实 MAE)
T2 × {S1, S2, S3} -> (相似度, 反事实 MAE)
T3 × {S1, S2, S3} -> (相似度, 反事实 MAE)
```

由此可以直接读出两种完全不同的失败模式：

- **相似度选错**：oracle 关系明显更好，但相似度没选中它（routing_regret 大、
  top1_hit=false）。修的是检索。
- **三条都不好**：即使 oracle 关系也打不过 seasonal_naive / validation_best_single。
  修的是候选池和组合本身，换检索没有用。

本脚本**只读**：不调用任何写库接口，不记录 prediction run，不改动冻结的数据库、
窗口计划或建库产物。它复用 Stage 2 的窗口切片、候选轨迹与基线冻结实现，保证与门控
用的是同一套口径。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage2_quality_gate import (
    BASELINE_BEST_SINGLE,
    BASELINE_SEASONAL_NAIVE,
    SCENARIO_SAMPLES,
    TEST_WINDOWS,
    Stage2Error,
    _candidate_matrix,
    _cross_check_library_report,
    _freeze_baselines,
    _window_slice,
    mae,
    rmse,
)
from scripts.train_combinations_kg import (
    MODEL_LIBRARY_BASE_HORIZON,
    MODEL_LIBRARY_BUSINESS_DOMAIN,
    MODEL_LIBRARY_COUNTRY_BY_REGION,
    MODEL_LIBRARY_TASK_TYPE,
    _frozen_windows,
    _library_candidate_models,
    _library_raw_frame,
    history_window_signature,
)
from src.core.index import ScenarioIndex
from src.models.artifacts import load_artifact
from src.models.trajectory_forecast import calendar_frame, generate_member_trajectory
from src.storage.model_store import SUPPORTED_FORECAST_STEPS, ModelStore


def _history_relations(
    store: ModelStore,
    report: Mapping[str, Any],
    *,
    dataset: str,
    forecast_steps: int,
) -> List[Dict[str, Any]]:
    """按 S1—S3 取回建库时写下的三条关系及其组合器产物。"""
    relations = []
    for window in SCENARIO_SAMPLES:
        task = next(
            t for t in report["tasks"]
            if t.get("dataset") == dataset
            and int(t.get("forecast_steps", -1)) == int(forecast_steps)
            and t.get("scenario_sample") == window
        )
        profile = store.get_data_profile(task["data_profile_id"])
        combination = store.get_combination(task["combination_id"])
        relations.append({
            "window": window,
            "scenario_id": task["scenario_id"],
            "relation_id": task["relation_id"],
            "data_profile_id": task["data_profile_id"],
            "data_ref": profile["data_ref"],
            "combination_id": task["combination_id"],
            # 与在线路径同口径：排序依据是数据画像的 signature，不是场景 signature
            "signature": profile["signature"],
            "members": [m["model_id"] for m in combination["members"]],
            "weights": [float(m["weight"]) for m in combination["members"]],
            "has_interaction": bool(task["has_interaction"]),
            "validation_mae": float(task["validation_mae"]),
            "predictor": load_artifact(Path(combination["artifact_path"])),
        })
    return relations


def _similarity(query_signature: Mapping[str, float], relations: Sequence[Mapping[str, Any]]):
    """用在线检索同一套 ScenarioIndex 打分，且同样按**数据画像**排序。

    键取 relation_id：同一个场景下可以有多条关系，用 scenario_id 当键会把它们折叠。
    """
    index = ScenarioIndex()
    for relation in relations:
        index.add({
            "scenario_id": str(relation["relation_id"]),
            "signature": relation["signature"],
            "business_domain": MODEL_LIBRARY_BUSINESS_DOMAIN,
        })
    ranked = index.query(signature=dict(query_signature))
    return (
        {int(row["scenario_id"]): float(row["_score"]) for row in ranked},
        int(ranked[0]["scenario_id"]),
    )


def attribute_task(
    *,
    store: ModelStore,
    raw_root: Path,
    dataset: str,
    forecast_steps: int,
    windows: Mapping[str, Mapping[str, Any]],
    declared_candidates: Sequence[str],
    library_report: Mapping[str, Any],
) -> Dict[str, Any]:
    country = MODEL_LIBRARY_COUNTRY_BY_REGION.get(dataset)
    if country is None:
        raise Stage2Error(f"未知数据集 {dataset}：无法确定日历特征所用节假日日历")
    raw = _library_raw_frame(raw_root, dataset)
    members, _skipped = _library_candidate_models(
        store, dataset=dataset, base_horizon=MODEL_LIBRARY_BASE_HORIZON,
        model_types=declared_candidates,
    )
    relations = _history_relations(
        store, library_report, dataset=dataset, forecast_steps=forecast_steps
    )

    frozen = _freeze_baselines(
        members, raw=raw, windows=windows, forecast_steps=forecast_steps,
        country=country, dataset=dataset,
    )
    _cross_check_library_report(
        library_report, dataset=dataset, forecast_steps=forecast_steps,
        declared_candidates=declared_candidates,
        validation_window_mae=frozen["validation_window_mae"],
    )
    columns = frozen["columns"]

    queries: List[Dict[str, Any]] = []
    for label in TEST_WINDOWS:
        history, target = _window_slice(
            raw, windows[label], forecast_steps, label=f"{dataset} {label}"
        )
        y = target["load"].to_numpy(dtype=float)
        matrix = _candidate_matrix(
            members, history, forecast_steps=forecast_steps, country=country,
            label=f"{dataset} {label}", required_columns=columns,
        )
        calendar = calendar_frame(target["timestamp"], country)
        signature = history_window_signature(
            history, freq="h", base_horizon=MODEL_LIBRARY_BASE_HORIZON
        )
        scores, selected_relation_id = _similarity(signature, relations)

        per_history = []
        for relation in relations:
            predictor = relation["predictor"]
            missing = [m for m in predictor.member_ids if m not in matrix.columns]
            if missing:
                raise Stage2Error(
                    f"{dataset} {label}: 关系 {relation['window']} 的成员 {missing} "
                    "在该窗口产不出轨迹，无法做反事实评价"
                )
            yhat = np.asarray(
                predictor.predict(
                    {m: matrix[m].to_numpy(dtype=float) for m in predictor.member_ids},
                    calendar,
                ),
                dtype=float,
            )
            per_history.append({
                "window": relation["window"],
                "scenario_id": relation["scenario_id"],
                "relation_id": relation["relation_id"],
                "data_profile_id": relation["data_profile_id"],
                "data_ref": relation["data_ref"],
                "members": [m.split("__")[-1] for m in relation["members"]],
                "weights": relation["weights"],
                "has_interaction": relation["has_interaction"],
                "validation_mae": relation["validation_mae"],
                "similarity": scores[relation["relation_id"]],
                "counterfactual_mae": mae(y, yhat),
                "counterfactual_rmse": rmse(y, yhat),
            })

        selected = next(e for e in per_history if e["relation_id"] == selected_relation_id)
        oracle = min(per_history, key=lambda e: (e["counterfactual_mae"], e["window"]))
        # 并列时 top1 必须算命中：多条关系可能给出完全相同的轨迹（例如都被归约成
        # 同一个单模型），此时按窗口名 tie-break 选出的 oracle 只是个记号，选中另一条
        # 并不是"选错"。命中判据取"是否达到了 oracle 的 MAE"，与 regret=0 自洽。
        oracle_windows = [
            e["window"] for e in per_history
            if np.isclose(e["counterfactual_mae"], oracle["counterfactual_mae"],
                          rtol=1e-12, atol=0.0)
        ]
        seasonal = generate_member_trajectory(
            model=None, model_type="seasonal_naive", required_features=[],
            history=history, forecast_steps=forecast_steps, country=country,
        )
        reference = {
            BASELINE_SEASONAL_NAIVE: mae(y, seasonal),
            BASELINE_BEST_SINGLE: mae(
                y, matrix[frozen["best_single"]].to_numpy(dtype=float)
            ),
        }
        best_reference = min(reference.values())
        queries.append({
            "window": label,
            "forecast_origin": str(windows[label]["forecast_origin"]),
            "query_signature": signature,
            "per_history_relation": per_history,
            "selected": {
                "window": selected["window"], "scenario_id": selected["scenario_id"],
                "relation_id": selected["relation_id"],
                "data_profile_id": selected["data_profile_id"],
                "similarity": selected["similarity"],
                "counterfactual_mae": selected["counterfactual_mae"],
            },
            "oracle": {
                "window": oracle["window"], "counterfactual_mae": oracle["counterfactual_mae"],
                "tied_windows": oracle_windows,
            },
            "routing_regret": (
                (selected["counterfactual_mae"] - oracle["counterfactual_mae"])
                / oracle["counterfactual_mae"]
            ),
            "top1_hit": selected["window"] in oracle_windows,
            "baseline_reference": reference,
            # oracle 都打不过基线 -> 问题不在检索，换关系没有用
            "oracle_beats_best_reference": oracle["counterfactual_mae"] < best_reference,
        })

    return {
        "dataset": dataset,
        "forecast_steps": int(forecast_steps),
        "validation_best_single": frozen["best_single"],
        "candidate_columns": columns,
        "history_relations": [
            {k: v for k, v in relation.items() if k != "predictor"}
            for relation in relations
        ],
        "queries": queries,
        "summary": {
            "top1_hit_rate": float(np.mean([q["top1_hit"] for q in queries])),
            "mean_routing_regret": float(np.mean([q["routing_regret"] for q in queries])),
            "queries_where_oracle_beats_best_reference": int(
                sum(q["oracle_beats_best_reference"] for q in queries)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 路由归因（只读）")
    parser.add_argument("--database", type=Path, required=True, help="冻结的 V3 SQLite 模型库")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--window-plan", type=Path, required=True)
    parser.add_argument("--library-report", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--forecast-steps", nargs="+", type=int,
                        default=list(SUPPORTED_FORECAST_STEPS))
    parser.add_argument("--candidates", nargs="+", required=True,
                        help="本批声明的候选模型类型，必须与建库时一致")
    parser.add_argument("--out", type=Path, required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    library_report = json.loads(args.library_report.read_text(encoding="utf-8"))
    store = ModelStore(str(args.database))
    tasks: List[Dict[str, Any]] = []
    try:
        for dataset in args.datasets:
            for steps in args.forecast_steps:
                tasks.append(
                    attribute_task(
                        store=store, raw_root=args.raw_root, dataset=dataset,
                        forecast_steps=int(steps),
                        windows=_frozen_windows(args.window_plan, dataset, int(steps)),
                        declared_candidates=args.candidates,
                        library_report=library_report,
                    )
                )
    except Stage2Error as exc:
        print(f"[attribution] 运行不完整，不产出归因结论: {exc}")
        return 1
    finally:
        store.close()

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"stage": "stage2_routing_attribution", "tasks": tasks},
                   indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    for task in tasks:
        print(f"\n[attribution] {task['dataset']} s={task['forecast_steps']}"
              f"（validation best single = {task['validation_best_single']}）")
        for query in task["queries"]:
            cells = "  ".join(
                f"{e['window']}: sim={e['similarity']:.4f} mae={e['counterfactual_mae']:.4f}"
                for e in query["per_history_relation"]
            )
            print(f"  {query['window']}  {cells}")
            print(f"    选中 {query['selected']['window']}，oracle "
                  f"{'/'.join(query['oracle']['tied_windows'])}，"
                  f"regret={query['routing_regret']:.4f}，top1_hit={query['top1_hit']}；"
                  f"基线 naive={query['baseline_reference'][BASELINE_SEASONAL_NAIVE]:.4f} "
                  f"best_single={query['baseline_reference'][BASELINE_BEST_SINGLE]:.4f}，"
                  f"oracle 胜过最好基线={query['oracle_beats_best_reference']}")
        summary = task["summary"]
        print(f"  小结: top1 命中率={summary['top1_hit_rate']:.3f}，"
              f"平均 routing regret={summary['mean_routing_regret']:.4f}，"
              f"oracle 胜过最好基线的窗口数="
              f"{summary['queries_where_oracle_beats_best_reference']}/{len(task['queries'])}")
    print(f"\n[attribution] 归因已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
