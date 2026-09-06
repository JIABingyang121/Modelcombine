"""Stage 2 路由归因报告（scripts/stage2_routing_attribution.py）。

这份报告要回答的是"相似度选错了"还是"三条历史组合本身都不好"，所以必须钉死三件事：
每个查询窗口对三条历史关系都给出相似度与反事实 MAE；oracle/regret/top1 与这些数字
自洽；以及它是**只读**的——跑完不得改动冻结的数据库。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.stage2_routing_attribution import attribute_task
from tests.forecast_steps_fixtures import DATASET, FIXTURE_CANDIDATES, REPO_ROOT
from tests.test_stage2_quality_gate import STEPS, _build_stage1


def _run(tmp_path: Path, built: dict, *, out_name="attribution.json", candidates=None):
    out = tmp_path / out_name
    proc = subprocess.run(
        [
            sys.executable, "scripts/stage2_routing_attribution.py",
            "--database", str(built["db"]), "--raw-root", str(built["raw_root"]),
            "--window-plan", str(built["window_plan"]),
            "--library-report", str(built["library_report"]),
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *(candidates or FIXTURE_CANDIDATES),
            "--out", str(out),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return proc, out


@pytest.fixture(scope="module")
def attribution(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("attribution")
    built = _build_stage1(tmp_path)
    db_before = built["db"].read_bytes()
    proc, out = _run(tmp_path, built)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return {
        "built": built, "proc": proc, "db_before": db_before,
        "report": json.loads(out.read_text()),
    }


def test_every_query_is_scored_against_all_three_history_relations(attribution):
    task = attribution["report"]["tasks"][0]
    assert [q["window"] for q in task["queries"]] == ["Q1", "Q2", "Q3"]
    assert [r["window"] for r in task["history_relations"]] == ["H1", "H2", "H3"]

    for query in task["queries"]:
        per_history = query["per_history_relation"]
        assert [e["window"] for e in per_history] == ["H1", "H2", "H3"]
        for entry in per_history:
            assert 0.0 <= entry["similarity"] <= 1.0
            assert np.isfinite(entry["counterfactual_mae"]) and entry["counterfactual_mae"] > 0
            assert np.isfinite(entry["counterfactual_rmse"])
            assert entry["members"] and entry["weights"]


def test_selected_oracle_regret_and_hit_rate_are_self_consistent(attribution):
    """选中项必须是相似度最高的那条；oracle 必须是反事实 MAE 最小的那条。"""
    task = attribution["report"]["tasks"][0]
    for query in task["queries"]:
        per_history = query["per_history_relation"]
        by_similarity = max(per_history, key=lambda e: e["similarity"])
        by_mae = min(per_history, key=lambda e: (e["counterfactual_mae"], e["window"]))

        assert query["selected"]["window"] == by_similarity["window"]
        assert query["selected"]["counterfactual_mae"] == pytest.approx(
            by_similarity["counterfactual_mae"], rel=1e-12
        )
        assert query["oracle"]["window"] == by_mae["window"]
        # 并列关系必须都被列进 tied_windows，且 top1 按"是否达到 oracle MAE"判定
        tied = {
            e["window"] for e in per_history
            if e["counterfactual_mae"] == pytest.approx(
                by_mae["counterfactual_mae"], rel=1e-12
            )
        }
        assert set(query["oracle"]["tied_windows"]) == tied
        expected_regret = (
            (query["selected"]["counterfactual_mae"] - by_mae["counterfactual_mae"])
            / by_mae["counterfactual_mae"]
        )
        assert query["routing_regret"] == pytest.approx(expected_regret, rel=1e-12)
        assert query["routing_regret"] >= -1e-12
        assert query["top1_hit"] is (query["selected"]["window"] in tied)
        # regret 为 0 与 top1 命中必须同进同出，不能出现"regret=0 但判未命中"
        assert query["top1_hit"] is bool(query["routing_regret"] <= 1e-12)
        assert query["oracle_beats_best_reference"] is bool(
            by_mae["counterfactual_mae"] < min(query["baseline_reference"].values())
        )

    summary = task["summary"]
    assert summary["top1_hit_rate"] == pytest.approx(
        float(np.mean([q["top1_hit"] for q in task["queries"]]))
    )
    assert summary["mean_routing_regret"] == pytest.approx(
        float(np.mean([q["routing_regret"] for q in task["queries"]]))
    )


def test_counterfactual_mae_really_uses_that_relation_not_the_selected_one(attribution):
    """三条关系若成员/权重不同，反事实 MAE 就必须不同——否则是拿同一条算了三遍。"""
    task = attribution["report"]["tasks"][0]
    signatures = {
        (tuple(r["members"]), tuple(r["weights"]), r["has_interaction"])
        for r in task["history_relations"]
    }
    if len(signatures) == 1:
        pytest.skip("本装置三条历史关系完全相同，反事实差异无从体现")

    for query in task["queries"]:
        maes = {e["window"]: e["counterfactual_mae"] for e in query["per_history_relation"]}
        assert len(set(maes.values())) > 1, maes


def test_report_is_read_only_against_the_frozen_database(attribution):
    """归因不得写库：不记录 prediction run，不改 use_count/last_used_at。"""
    assert attribution["built"]["db"].read_bytes() == attribution["db_before"]


def test_baseline_reference_lets_us_tell_bad_routing_from_bad_relations(attribution):
    """每个查询都必须带上基线参照，否则无法区分两种失败模式。"""
    task = attribution["report"]["tasks"][0]
    for query in task["queries"]:
        reference = query["baseline_reference"]
        assert set(reference) == {"seasonal_naive_168", "validation_best_single"}
        assert all(np.isfinite(v) and v > 0 for v in reference.values())
    assert 0 <= task["summary"]["queries_where_oracle_beats_best_reference"] <= 3


def test_candidate_mismatch_is_incomplete_and_writes_no_report(tmp_path):
    built = _build_stage1(tmp_path)
    proc, out = _run(
        tmp_path, built, out_name="bad.json",
        candidates=[c for c in FIXTURE_CANDIDATES if c != "seasonal_naive"],
    )

    assert proc.returncode == 1
    assert "不是同一批候选" in (proc.stdout + proc.stderr)
    assert not out.exists()


def test_attribute_task_rejects_unknown_dataset(tmp_path):
    from scripts.stage2_quality_gate import Stage2Error
    from src.storage.model_store import ModelStore

    built = _build_stage1(tmp_path)
    store = ModelStore(str(built["db"]))
    try:
        with pytest.raises(Stage2Error, match="节假日日历"):
            attribute_task(
                store=store, raw_root=built["raw_root"], dataset="unknown_grid",
                forecast_steps=STEPS, windows={}, declared_candidates=FIXTURE_CANDIDATES,
                library_report={"tasks": []},
            )
    finally:
        store.close()
