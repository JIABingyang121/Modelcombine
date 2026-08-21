"""Task 8.2：Task 7 基线训练必须与默认 Protocol B 候选配置对齐。"""
from __future__ import annotations

import hashlib
import json

import pytest

import scripts.train_baselines as baselines


def test_build_model_forwards_pipeline_parameters_and_preserves_arima_frequency(monkeypatch):
    """若基线训练丢弃 pipeline 参数，Task 7 就不能验证当前默认入口。"""
    calls = []

    def create(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return {"model_id": model_id, "kwargs": kwargs}

    monkeypatch.setattr(baselines.model_registry, "create", create)

    lgbm = baselines.build_model("lgbm_reg", "h", {"n_jobs": 1, "deterministic": True})
    arima = baselines.build_model("arima", "D", {"order": [1, 1, 1]})

    assert lgbm["kwargs"] == {"n_jobs": 1, "deterministic": True}
    assert arima["kwargs"] == {"order": [1, 1, 1], "freq": "D"}
    assert calls == [
        ("lgbm_reg", {"n_jobs": 1, "deterministic": True}),
        ("arima", {"order": [1, 1, 1], "freq": "D"}),
    ]


def test_load_pipeline_model_params_returns_only_the_model_mapping(tmp_path):
    """基线只消费 models 配置；无关的 data 配置不得混入 registry.create。"""
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        "data:\n  test_days: 30\nmodels:\n  lgbm_reg:\n    n_jobs: 1\n",
        encoding="utf-8",
    )

    assert baselines.load_pipeline_model_params(pipeline) == {"lgbm_reg": {"n_jobs": 1}}


def test_baseline_provenance_binds_config_hash_and_candidate_seed_strategy(tmp_path):
    """若换了 pipeline.yaml 或跳过候选种子，v4 证据必须能识别出来。"""
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("models:\n  lgbm_reg:\n    n_jobs: 1\n", encoding="utf-8")

    provenance = baselines.build_baseline_provenance(
        pipeline_config=pipeline,
        model_params={"lgbm_reg": {"n_jobs": 1}},
        global_seed=42,
    )

    assert provenance["schema_version"] == 1
    assert provenance["pipeline_config"]["sha256"] == hashlib.sha256(pipeline.read_bytes()).hexdigest()
    assert provenance["candidate_seed_strategy"] == "sha256(global_seed|model_id|stage)"
    assert provenance["models"] == {"lgbm_reg": {"n_jobs": 1}}


def test_baseline_provenance_loader_rejects_pipeline_hash_mismatch(tmp_path):
    """Task 7 v4 不得把另一份 pipeline 配置产出的预测当成默认路径证据。"""
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("models:\n  lgbm_reg:\n    n_jobs: 1\n", encoding="utf-8")
    pred_root = tmp_path / "baselines"
    pred_root.mkdir()
    (pred_root / "baseline_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline_config": {"path": str(pipeline), "sha256": "wrong"},
                "candidate_seed_strategy": "sha256(global_seed|model_id|stage)",
                "global_seed": 42,
                "models": {"lgbm_reg": {"n_jobs": 1}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sha256"):
        baselines.load_verified_baseline_provenance(pred_root, pipeline)


def test_baseline_provenance_loader_rejects_a_different_global_seed(monkeypatch, tmp_path):
    """种子算法相同但基准种子不同，仍不是同一默认发布配置。"""
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("models:\n  lgbm_reg:\n    n_jobs: 1\n", encoding="utf-8")
    pred_root = tmp_path / "baselines"
    pred_root.mkdir()
    (pred_root / "baseline_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline_config": {
                    "path": str(pipeline),
                    "sha256": hashlib.sha256(pipeline.read_bytes()).hexdigest(),
                },
                "candidate_seed_strategy": "sha256(global_seed|model_id|stage)",
                "global_seed": 7,
                "models": {"lgbm_reg": {"n_jobs": 1}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(baselines, "global_seed", lambda: 42)

    with pytest.raises(ValueError, match="global_seed"):
        baselines.load_verified_baseline_provenance(pred_root, pipeline)


def test_run_dataset_records_the_per_task_seed_used_before_each_candidate_fit(monkeypatch, tmp_path):
    """若训练循环遗漏 seed_for_candidate，v4 的“可重复”来源清单就是空话。"""
    dataset_root = tmp_path / "features" / "pjm"
    dataset_root.mkdir(parents=True)
    for split in ("train", "val", "test"):
        frame = {
            "timestamp": ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00", "2026-01-01 03:00"],
            "load": [10.0, 11.0, 12.0, 13.0],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
        import pandas as pd

        pd.DataFrame(frame).to_csv(dataset_root / f"{split}.csv", index=False)

    seed_calls = []

    class ConstantModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [10.0] * len(X)

    monkeypatch.setattr(
        baselines,
        "seed_for_candidate",
        lambda model_id, stage: seed_calls.append((model_id, stage)) or 123,
    )
    monkeypatch.setattr(baselines, "build_model", lambda model_id, freq, params: ConstantModel())

    baselines.run_dataset(
        "pjm",
        dataset_root,
        "load",
        [1],
        tmp_path / "out",
        None,
        {"lgbm_reg": {"n_jobs": 1}},
    )

    expected_models = {
        "xgboost_reg", "lgbm_reg", "catboost_reg", "prophet", "arima", "power_difference", "multimodal_fusion"
    }
    assert set(seed_calls) == {(model_id, "baseline:pjm:h1:fit") for model_id in expected_models}
    meta = json.loads((tmp_path / "out" / "pjm" / "model_meta_h1_lgbm_reg.json").read_text(encoding="utf-8"))
    assert meta["training_seed"] == 123
    assert meta["candidate_seed_strategy"] == "sha256(global_seed|model_id|stage)"


def test_task_artifact_verification_rejects_a_prediction_not_bound_to_provenance(tmp_path):
    """旧 CSV 即使留在目录里，只要未被本轮清单绑定，就不得进入 v4 矩阵。"""
    pred_root = tmp_path / "baselines"
    dataset_dir = pred_root / "pjm"
    dataset_dir.mkdir(parents=True)
    files = {}
    for name in (
        "val_pred_h1_lgbm_reg.csv",
        "test_pred_h1_lgbm_reg.csv",
        "model_meta_h1_lgbm_reg.json",
    ):
        path = dataset_dir / name
        path.write_text(name, encoding="utf-8")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # 这是遗留的旧模型文件：存在但没有出现在 provenance 清单中。
    (dataset_dir / "val_pred_h1_xgboost_reg.csv").write_text("old", encoding="utf-8")
    (dataset_dir / "test_pred_h1_xgboost_reg.csv").write_text("old", encoding="utf-8")
    (dataset_dir / "model_meta_h1_xgboost_reg.json").write_text("old", encoding="utf-8")
    provenance = {
        "artifacts": [
            {
                "dataset": "pjm",
                "horizon": 1,
                "model": "lgbm_reg",
                "files": files,
            }
        ]
    }

    with pytest.raises(ValueError, match="xgboost_reg"):
        baselines.verify_task_artifacts(
            provenance,
            pred_root=pred_root,
            dataset="pjm",
            horizon=1,
            models=["lgbm_reg", "xgboost_reg"],
        )

    baselines.verify_task_artifacts(
        provenance,
        pred_root=pred_root,
        dataset="pjm",
        horizon=1,
        models=["lgbm_reg"],
    )
    (dataset_dir / "val_pred_h1_lgbm_reg.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="lgbm_reg"):
        baselines.verify_task_artifacts(
            provenance,
            pred_root=pred_root,
            dataset="pjm",
            horizon=1,
            models=["lgbm_reg"],
        )
