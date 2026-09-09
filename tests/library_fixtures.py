"""合成一个"完整模型库"，用来测只读预检本身。

预检只读数据库和建库报告，所以这里直接写行、不跑真实建库——真实建库的正确性由
`tests/test_offline_model_library_wiring.py` 与 `tests/test_formal_library_build.py` 覆盖。
这份装置的作用是能廉价地造出"完整"与各种"不完整"的库，看预检抓不抓得住。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from scripts.train_combinations_kg import TIMESTAMP_POLICY
from src.storage.model_store import ModelStore

SCENARIO_SAMPLES = ("S1", "S2", "S3")


def make_complete_library(
    root: Path,
    *,
    datasets: Sequence[str],
    forecast_steps: Sequence[int],
    candidates: Sequence[str],
    base_horizon: int = 1,
    name: str = "lib.sqlite3",
) -> dict:
    """写出一个内部完整的库：每个数据集×候选一个基础模型，每个数据集×长度三条关系。"""
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    database = root / name
    store = ModelStore(str(database))
    store.create_schema()

    model_ids = []
    for dataset in datasets:
        for model_type in candidates:
            model_id = f"{dataset}__h{base_horizon}__{model_type}"
            path = artifacts / f"{model_id}.pkl"
            path.write_bytes(b"model")
            store.add_model(
                model_id=model_id, model_type=model_type, task_type="load_forecast",
                artifact_path=str(path), required_features=["hour"],
                model_params={}, lifecycle_stage="active",
            )
            model_ids.append(model_id)

    tasks = []
    for dataset in datasets:
        for steps in forecast_steps:
            for index, sample in enumerate(SCENARIO_SAMPLES):
                scenario_id = f"{dataset}_h{base_horizon}_s{steps}_{sample}"
                store.add_scenario(
                    scenario_id=scenario_id, task_type="load_forecast",
                    business_domain="power", region=dataset, horizon=base_horizon,
                    forecast_steps=int(steps), freq="h", signature={"w": index},
                )
                profile = store.add_data_profile(
                    scenario_id=scenario_id, data_ref=f"/raw/{dataset}",
                    target_column="load", features=["hour"], sample_count=int(steps),
                    start_at="2026-01-01 00:00:00", end_at="2026-01-02 00:00:00",
                    signature={"w": index},
                )
                combo_path = artifacts / f"{scenario_id}__combo.pkl"
                combo_path.write_bytes(b"combo")
                combination = store.add_combination(
                    "protocol_b_combination", str(combo_path),
                    [(f"{dataset}__h{base_horizon}__{candidates[0]}", 0, 1.0)],
                )
                store.add_relation(scenario_id, profile, combination, validation_mae=1.0)
                tasks.append({
                    "dataset": dataset, "forecast_steps": int(steps),
                    "scenario_sample": sample, "scenario_id": scenario_id,
                })
    store.close()

    report = root / "model_library_report.json"
    report.write_text(json.dumps({
        "base_horizon": base_horizon, "signature_window": 720,
        "timestamp_policy": TIMESTAMP_POLICY,
        "candidate_model_types": list(candidates), "tasks": tasks,
    }), encoding="utf-8")

    pipeline = root / "pipeline.yaml"
    pipeline.write_text(
        "models:\n" + "".join(f"  {c}:\n    alpha: 1\n" for c in candidates),
        encoding="utf-8",
    )
    return {
        "database": database, "artifacts": artifacts, "report": report,
        "pipeline": pipeline, "model_ids": model_ids, "tasks": tasks,
    }
