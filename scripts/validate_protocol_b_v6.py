"""Protocol B v6 两轮验收验证器（Task 8.3 Task 7）。

消费两份 `task8-v6.1` shadow 报告与候选诊断报告，校验两轮一致性后写出
`v6_acceptance.json`（core_status 与 relation_qualification_status 分开）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from scripts.run_protocol_b_candidate_diagnostic import validate_diagnostic_schema
from scripts.run_system_ab_shadow import (
    evaluate_relation_qualification,
    evaluate_v6_core_gates,
    validate_shadow_report as validate_shadow_report_impl,
)


def _tasks(report: Mapping[str, Any]) -> list:
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("report.tasks must be a list")
    return tasks


def _task_key(task: Mapping[str, Any]) -> tuple:
    return (task.get("dataset"), task.get("horizon"))


def _tasks_by_key(report: Mapping[str, Any]) -> dict:
    return {_task_key(t): t for t in _tasks(report)}


def assert_same_task_models(run1: Mapping[str, Any], run2: Mapping[str, Any]) -> None:
    """两轮各任务的最终模型集合必须一致（顺序无关，按任务键匹配）。"""
    t1, t2 = _tasks_by_key(run1), _tasks_by_key(run2)
    if set(t1) != set(t2):
        raise ValueError(f"run1/run2 task keys differ: {sorted(set(t1) ^ set(t2))}")
    for key in t1:
        a, b = t1[key], t2[key]
        if set(a.get("selected_models") or []) != set(b.get("selected_models") or []):
            raise ValueError(
                f"task {key}: selected_models differ "
                f"{a.get('selected_models')} vs {b.get('selected_models')}"
            )


def _compare_mapping(va: Any, vb: Any, *, label: str, atol: float) -> None:
    if set(va) != set(vb):
        raise ValueError(f"{label}: keys differ")
    for k in va:
        if abs(float(va[k]) - float(vb[k])) > atol:
            raise ValueError(f"{label}.{k} differs {va[k]} vs {vb[k]}")


def assert_numeric_close(
    run1: Mapping[str, Any],
    run2: Mapping[str, Any],
    *,
    fields: Sequence[str] = ("weights", "test_mae_on"),
    atol: float = 1e-8,
) -> None:
    """指定字段在两轮之间逐任务 1e-8 内一致。

    "weights" 特殊处理：比较真实的 `protocol_b_on.weights` 与 `protocol_b_off.weights`。
    """
    t1, t2 = _tasks_by_key(run1), _tasks_by_key(run2)
    if set(t1) != set(t2):
        raise ValueError(f"run1/run2 task keys differ: {sorted(set(t1) ^ set(t2))}")
    for key in t1:
        a, b = t1[key], t2[key]
        label = f"{key}"
        for field in fields:
            if field == "weights":
                for arm in ("protocol_b_on", "protocol_b_off"):
                    wa = (a.get(arm) or {}).get("weights") if isinstance(a.get(arm), Mapping) else {}
                    wb = (b.get(arm) or {}).get("weights") if isinstance(b.get(arm), Mapping) else {}
                    if not isinstance(wa, Mapping) or not isinstance(wb, Mapping):
                        raise ValueError(f"{label}: {arm}.weights missing")
                    _compare_mapping(wa, wb, label=f"{label}.{arm}.weights", atol=atol)
            else:
                va, vb = a.get(field), b.get(field)
                if isinstance(va, Mapping) and isinstance(vb, Mapping):
                    _compare_mapping(va, vb, label=f"{label}.{field}", atol=atol)
                elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    if abs(float(va) - float(vb)) > atol:
                        raise ValueError(f"{label}: {field} differs {va} vs {vb}")
                else:
                    if va != vb:
                        raise ValueError(f"{label}: {field} differs {va} vs {vb}")


def assert_same_locked_sources(run1: Mapping[str, Any], run2: Mapping[str, Any]) -> None:
    """两轮必须消费同一份锁定数据哈希与来源。"""
    m1 = run1.get("_meta") or {}
    m2 = run2.get("_meta") or {}
    for key in ("data_hashes", "baseline_provenance", "pipeline_config"):
        if m1.get(key) != m2.get(key):
            raise ValueError(f"_meta.{key} differs between runs")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_flow_from_traces(trace_root: Path) -> dict:
    flows = {}
    for dataset in ("pjm", "aemo_vic", "aemo_nsw"):
        for horizon in (1, 6, 24):
            path = trace_root / dataset / f"h{horizon}" / "protocol_b_trace_on.json"
            if not path.exists():
                raise ValueError(f"missing selection-flow trace: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            flow = None
            for stage in payload.get("stages", []):
                if isinstance(stage, Mapping) and stage.get("stage") == "ProtocolBBackend":
                    outputs = stage.get("outputs") or {}
                    flow = outputs.get("selection_flow")
            if not isinstance(flow, Mapping):
                raise ValueError(f"missing or empty selection_flow in {path}")
            flows[f"{dataset}_h{horizon}"] = flow
    if len(flows) != 9:
        raise ValueError(f"expected 9 selection flows, got {len(flows)}")
    return flows


def assert_same_selection_flow(run1_trace_root: Path, run2_trace_root: Path) -> None:
    f1 = _selection_flow_from_traces(run1_trace_root)
    f2 = _selection_flow_from_traces(run2_trace_root)
    if f1 != f2:
        raise ValueError("selection_flow differs between runs")


def assert_pair_references_match_diagnostic(
    run: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> None:
    """报告里的 pair 基准必须逐任务来自候选诊断文件，不能是凭空编造的。

    只有 `best_pair_reference`（静态诊断基准）必须与诊断一致。`best_pair_same_as_protocol_b`
    与 `explicit_conflict_edges_consumed` 是**本轮运行时**结果，不能强制等于静态诊断的旧值——
    否则关系机制真的改变了本轮模型组合、或本轮真实触发显式 conflict 时验收反而失败。
    """
    diag_tasks = {_task_key(t): t for t in (diagnostic.get("tasks") or [])}
    for task in _tasks(run):
        key = _task_key(task)
        dt = diag_tasks.get(key)
        if dt is None:
            raise ValueError(f"task {key}: missing candidate diagnostic entry")
        if task.get("best_pair_reference") != dt.get("best_pair"):
            raise ValueError(f"task {key}: best_pair_reference differs from diagnostic")
        # best_pair_same_as_protocol_b 按本轮 selected_models 重算并核对，
        # 不强制等于静态诊断的 best_pair_same_as_protocol_b。
        bp_models = set((dt.get("best_pair") or {}).get("models") or [])
        expected_same = bool(bp_models and bp_models == set(task.get("selected_models") or []))
        if bool(task.get("best_pair_same_as_protocol_b")) != expected_same:
            raise ValueError(
                f"task {key}: best_pair_same_as_protocol_b does not match selected_models"
            )
        # explicit_conflict_edges_consumed 保留本轮实际值，不与静态诊断强制相等。


def assert_pair_runtime_consistent(
    run1: Mapping[str, Any],
    run2: Mapping[str, Any],
) -> None:
    """两轮运行时 pair 重合与显式 conflict 消费必须逐任务一致。

    v6_acceptance 用 run1 生成最终运行时汇总，因此 run2 的 `best_pair_same_as_protocol_b`
    与 `explicit_conflict_edges_consumed` 必须与 run1 一致，否则汇总失去意义。
    """
    t1, t2 = _tasks_by_key(run1), _tasks_by_key(run2)
    if set(t1) != set(t2):
        raise ValueError(f"run1/run2 task keys differ: {sorted(set(t1) ^ set(t2))}")
    for key in t1:
        a, b = t1[key], t2[key]
        if bool(a.get("best_pair_same_as_protocol_b")) != bool(b.get("best_pair_same_as_protocol_b")):
            raise ValueError(f"task {key}: best_pair_same_as_protocol_b differs between runs")
        if int(a.get("explicit_conflict_edges_consumed") or 0) != int(
            b.get("explicit_conflict_edges_consumed") or 0
        ):
            raise ValueError(f"task {key}: explicit_conflict_edges_consumed differs between runs")


def build_v6_acceptance(
    run1: Mapping[str, Any],
    run2: Mapping[str, Any],
    *,
    run1_trace_root: Path,
    run2_trace_root: Path,
    diagnostic: Optional[Mapping[str, Any]] = None,
) -> dict:
    validate_shadow_report_impl(run1, trace_root=run1_trace_root)
    validate_shadow_report_impl(run2, trace_root=run2_trace_root)
    assert_same_locked_sources(run1, run2)
    assert_same_task_models(run1, run2)
    assert_numeric_close(run1, run2, fields=("weights", "test_mae_on"), atol=1e-8)
    assert_same_selection_flow(run1_trace_root, run2_trace_root)
    if diagnostic is not None:
        # 完整执行诊断 schema 校验（不只比对版本字符串），并让两份报告都绑定同一诊断。
        validate_diagnostic_schema(diagnostic)
        assert_pair_references_match_diagnostic(run1, diagnostic)
        assert_pair_references_match_diagnostic(run2, diagnostic)
        assert_pair_runtime_consistent(run1, run2)

    tasks = _tasks(run1)
    core = evaluate_v6_core_gates(tasks, trace_root=run1_trace_root)
    relation = evaluate_relation_qualification(tasks)

    # 确定性输出：只存内容哈希与相对引用，不存时间戳/绝对路径。
    return {
        "schema_version": "v6-acceptance.1",
        "core_status": core.get("status"),
        "core_gates": core,
        "relation_qualification_status": relation.get("status"),
        "relation_qualification": relation,
    }


def _resolve_trace_root(meta: Mapping[str, Any], report_path: Path) -> Path:
    """优先显式传入的 trace root；否则回退到报告旁的相对目录。

    服务器 `_meta.out_root` 是绝对路径，拉回本机后不存在，不能直接用；此时应
    使用报告旁的 `traces/` 目录，保证本机与服务器重算一致。
    """
    out_root = meta.get("out_root")
    if out_root:
        candidate = Path(str(out_root))
        if candidate.exists():
            return candidate
    traces = report_path.parent / "traces"
    if traces.exists():
        return traces
    return report_path.parent


def _runtime_pair_overlap_and_conflict(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """从本轮任务记录汇总 pair 重合与显式 conflict 消费（不读静态诊断 summary）。"""
    overlap_tasks = [
        f"{t.get('dataset')}_h{t.get('horizon')}"
        for t in tasks
        if t.get("best_pair_same_as_protocol_b")
    ]
    conflict_tasks = [
        f"{t.get('dataset')}_h{t.get('horizon')}"
        for t in tasks
        if int(t.get("explicit_conflict_edges_consumed") or 0) > 0
    ]
    return {
        "pair_reference_overlap": {"count": len(overlap_tasks), "tasks": overlap_tasks},
        "explicit_conflict_edges_consumed": {
            "count": len(conflict_tasks), "tasks": conflict_tasks,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--candidate-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run1-trace-root", type=Path, default=None)
    parser.add_argument("--run2-trace-root", type=Path, default=None)
    args = parser.parse_args()

    run1 = json.loads(args.run1.read_text(encoding="utf-8"))
    run2 = json.loads(args.run2.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.candidate_diagnostic.read_text(encoding="utf-8"))

    # 两轮报告的 candidate_diagnostic_sha256 必须与本诊断文件一致。
    diag_sha = _sha256_file(args.candidate_diagnostic)
    for name, run in (("run1", run1), ("run2", run2)):
        meta = run.get("_meta") or {}
        if meta.get("candidate_diagnostic_sha256") != diag_sha:
            raise ValueError(f"{name} candidate_diagnostic_sha256 mismatch")

    meta1 = run1.get("_meta") or {}
    meta2 = run2.get("_meta") or {}
    trace_root1 = args.run1_trace_root or _resolve_trace_root(meta1, args.run1)
    trace_root2 = args.run2_trace_root or _resolve_trace_root(meta2, args.run2)

    acceptance = build_v6_acceptance(
        run1, run2,
        run1_trace_root=trace_root1,
        run2_trace_root=trace_root2,
        diagnostic=diagnostic,
    )
    acceptance["run1_report_sha256"] = _sha256_file(args.run1)
    acceptance["run2_report_sha256"] = _sha256_file(args.run2)
    acceptance["candidate_diagnostic_sha256"] = _sha256_file(args.candidate_diagnostic)
    # 本轮 v6 激活情况必须从 run1 的任务记录汇总，静态诊断 summary 另起字段保留，
    # 不冒充本轮实际触发情况。
    runtime = _runtime_pair_overlap_and_conflict(_tasks(run1))
    acceptance["pair_reference_overlap"] = runtime["pair_reference_overlap"]
    acceptance["explicit_conflict_edges_consumed"] = runtime["explicit_conflict_edges_consumed"]
    acceptance["candidate_diagnostic_summary"] = diagnostic.get("summary") or {}

    args.output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"core_status={acceptance['core_status']} "
          f"relation_qualification_status={acceptance['relation_qualification_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
