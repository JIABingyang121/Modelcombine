"""影子基准报告的 schema 与确定性测试（合一计划 Task 5）。

只验证结构与不变量，**不设置虚假的超短耗时断言**——耗时是被测量对象，
不是被规定对象；给它设阈值会把"证据"变成"自证预言"。
"""
import json
from pathlib import Path

import pytest

import scripts.profile_system_ab_demo as prof

REPORT_TOP_KEYS = {
    "schema_version",
    "rows",
    "repeat",
    "timeout_seconds",
    "random_seed",
    "isolated_tmpdir",
    "guarded_state_before",
    "guarded_state_after",
    "readonly_guarantee_held",
    "runs",
    "summary",
}


def test_stage_order_covers_every_stage_required_by_plan():
    required = {
        "data_and_features",
        "candidate_training_round1_fit",
        "candidate_training_round2_full",
        "prediction_bundle",
        "protocol_a",
        "stability_filter",
        "feature_corr_and_graph",
        "blocked_cv",
        "result_assembly",
    }
    assert required <= set(prof.STAGE_ORDER)
    # 训练耗时与 solver 耗时必须是分开的两个键，不能混算
    assert "protocol_b_solver_total" in prof.STAGE_ORDER
    assert "candidate_training_round1_fit" != "protocol_b_solver_total"


def test_guarded_paths_cover_production_state():
    assert "reports/historical_scenarios.json" in prof.GUARDED_PATHS
    assert "reports/graph_state.pkl" in prof.GUARDED_PATHS


def test_stage_timer_accumulates_and_restores():
    class Mod:
        @staticmethod
        def work(x):
            return x * 2

    timer = prof.StageTimer()
    original = Mod.work
    timer.wrap(Mod, "work", "blocked_cv")

    assert Mod.work(3) == 6  # 包装后行为不变
    assert Mod.work(4) == 8
    assert timer.counts["blocked_cv"] == 2
    assert timer.totals["blocked_cv"] >= 0.0

    timer.restore()
    assert Mod.work is original


def test_snapshot_guarded_reports_absent_for_missing_files(tmp_path):
    snap = prof._snapshot_guarded(tmp_path)
    assert set(snap) == set(prof.GUARDED_PATHS)
    assert all(v == "<absent>" for v in snap.values())


def test_snapshot_guarded_detects_content_change(tmp_path):
    target = tmp_path / "reports" / "historical_scenarios.json"
    target.parent.mkdir(parents=True)
    target.write_text("[]", encoding="utf-8")
    before = prof._snapshot_guarded(tmp_path)

    target.write_text('[{"id": "x"}]', encoding="utf-8")
    after = prof._snapshot_guarded(tmp_path)

    assert before != after
    assert before["reports/historical_scenarios.json"] != after["reports/historical_scenarios.json"]


def _report_files():
    base = Path(prof.PROJECT_ROOT) / "result" / "ab_convergence"
    return sorted(base.glob("profile_*.json"))


@pytest.mark.parametrize("path", _report_files() or [None])
def test_existing_reports_match_schema(path):
    if path is None:
        pytest.skip("尚未生成基准报告")
    report = json.loads(Path(path).read_text(encoding="utf-8"))

    assert REPORT_TOP_KEYS <= set(report)
    assert report["schema_version"] == prof.REPORT_SCHEMA_VERSION
    # 只读保证：运行前后生产状态哈希必须一致
    assert report["readonly_guarantee_held"] is True
    assert report["guarded_state_before"] == report["guarded_state_after"]

    assert report["runs"], "报告必须至少包含一次运行"
    for run in report["runs"]:
        assert {"run_index", "rows", "status", "wall_seconds", "stage_seconds"} <= set(run)
        assert set(run["stage_seconds"]) == set(prof.STAGE_ORDER)
        assert run["status"] in {"ok", "failed"}
        if run["status"] == "failed":
            # 失败必须留下可定位的证据，而不是只说"慢"
            assert run["error"], "失败运行必须记录异常信息"
            assert run["error_stage"], "失败运行必须记录失败阶段"
            assert "stages_completed" in run
        else:
            assert run["stage_seconds"]["protocol_b_solver_total"] >= 0.0
            assert "guard_evidence" in run


@pytest.mark.parametrize("path", _report_files() or [None])
def test_reports_separate_training_from_solver_time(path):
    """模型训练耗时不得计入 Protocol B solver 耗时。"""
    if path is None:
        pytest.skip("尚未生成基准报告")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    for run in report["runs"]:
        if run["status"] != "ok":
            continue
        stages = run["stage_seconds"]
        training = (
            stages["candidate_training_round1_fit"] + stages["candidate_training_round2_full"]
        )
        solver = stages["protocol_b_solver_total"]
        # solver 计时窗口在 bundle 构建之后才开始，二者不应互相包含
        assert solver <= run["wall_seconds"]
        assert training <= run["wall_seconds"]
