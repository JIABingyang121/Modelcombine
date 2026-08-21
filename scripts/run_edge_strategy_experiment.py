"""三策略对照实验驱动脚本（总纲 §8.6）。

三组：fixed / moving_average / hawkes（MODELCOMBINE_PIPELINE_EDGE_UPDATE_STRATEGY）。
隔离：每组独立 graph_state（MODELCOMBINE_GRAPH_STATE_PATH）+ 每组开跑前恢复同一份 reports/ 基线快照。
注入：第 INJECT_ROUND 轮把测试窗（最后 TEST_ROWS 行）负荷放大 INJECT_SCALE 倍，该轮结束后还原。

用法：
  venv/bin/python scripts/run_edge_strategy_experiment.py prepare   # 建基线快照+备份数据
  venv/bin/python scripts/run_edge_strategy_experiment.py run --arm fixed [--rounds 9]
  venv/bin/python scripts/run_edge_strategy_experiment.py run --all [--rounds 9]
  venv/bin/python scripts/run_edge_strategy_experiment.py summarize # 汇总 summary.json
  venv/bin/python scripts/run_edge_strategy_experiment.py restore   # 恢复 reports/ 与数据
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "result" / "0716_edge_strategy"
BASELINE = EXP_DIR / "baseline_snapshot"
ARMS_DIR = EXP_DIR / "arms"
REPORTS = ROOT / "reports"
DATA_CSV = ROOT / "data" / "pjm" / "load.csv"
DATA_BACKUP = EXP_DIR / "load.csv.orig"

ARMS = ("fixed", "moving_average", "hawkes")
DEFAULT_ROUNDS = 9
INJECT_ROUND = 5
INJECT_SCALE = 1.3
TEST_ROWS = 720  # run.py 的测试窗行数


def prepare() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    if BASELINE.exists():
        print(f"[skip] 基线快照已存在: {BASELINE}")
    else:
        shutil.copytree(REPORTS, BASELINE)
        # 所有组从全新图谱起步（公平起点，避免既有 hawkes 事件历史偏置）
        for stale in ("graph_state.pkl",):
            target = BASELINE / stale
            if target.exists():
                target.unlink()
        print(f"[ok] 基线快照: {BASELINE}")
    if not DATA_BACKUP.exists():
        shutil.copy2(DATA_CSV, DATA_BACKUP)
        print(f"[ok] 数据备份: {DATA_BACKUP}")
    for arm in ARMS:
        (ARMS_DIR / arm).mkdir(parents=True, exist_ok=True)


def _restore_reports_from_baseline() -> None:
    if not BASELINE.exists():
        raise SystemExit("先运行 prepare 建立基线快照")
    shutil.rmtree(REPORTS)
    shutil.copytree(BASELINE, REPORTS)


def _restore_data() -> None:
    if DATA_BACKUP.exists():
        shutil.copy2(DATA_BACKUP, DATA_CSV)


def _inject_drift() -> None:
    import pandas as pd

    df = pd.read_csv(DATA_BACKUP)
    df.loc[df.index[-TEST_ROWS:], "load"] = df["load"].iloc[-TEST_ROWS:] * INJECT_SCALE
    df.to_csv(DATA_CSV, index=False)
    print(f"[inject] 测试窗 {TEST_ROWS} 行负荷 ×{INJECT_SCALE}")


def _graph_edge_metrics(graph_path: Path) -> list:
    if not graph_path.exists():
        return []
    d = pickle.load(open(graph_path, "rb"))
    out = []
    for e in d.get("edges") or []:
        if not (isinstance(e, (list, tuple)) and len(e) == 3):
            continue
        u, v, data = e
        if isinstance(data, dict) and data.get("edge_type") == "recommended_for":
            out.append({
                "source": u, "target": v,
                "weight": data.get("weight"),
                "dynamic_strength": data.get("dynamic_strength"),
                "event_count": data.get("event_count"),
                "history_len": len(data.get("event_history") or []),
            })
    return out


def _latest_trace() -> dict:
    traces = sorted((REPORTS / "traces").glob("*.json"), key=os.path.getmtime)
    if not traces:
        return {}
    t = json.load(open(traces[-1]))
    path_id = None
    for s in t.get("stages", []):
        if s.get("stage") == "CombinatorBackend":
            path_id = (s.get("outputs") or {}).get("path_id")
    return {
        "path_id": path_id,
        "final_selection": t.get("final_selection"),
        "stages": [s["stage"] for s in t.get("stages", [])],
        "trace_file": traces[-1].name,
    }


def run_arm(arm: str, rounds: int) -> None:
    assert arm in ARMS, arm
    arm_dir = ARMS_DIR / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    graph_path = arm_dir / "graph_state.pkl"
    results_path = arm_dir / "rounds.jsonl"
    if results_path.exists():
        results_path.unlink()
    if graph_path.exists():
        graph_path.unlink()

    _restore_reports_from_baseline()
    _restore_data()

    env = dict(os.environ)
    env["MODELCOMBINE_PIPELINE_EDGE_UPDATE_STRATEGY"] = arm
    env["MODELCOMBINE_GRAPH_STATE_PATH"] = str(graph_path)
    env.pop("MODELCOMBINE_PIPELINE_ENABLE_TEMPORAL_RELATIONS", None)  # 策略开关全权接管

    for r in range(1, rounds + 1):
        injected = r == INJECT_ROUND
        try:
            if injected:
                _inject_drift()
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, str(ROOT / "run.py")],
                cwd=str(ROOT), env=env,
                capture_output=True, text=True, timeout=900,
            )
            wall = time.time() - t0
        finally:
            if injected:
                _restore_data()

        report = {}
        report_path = REPORTS / "report.json"
        if report_path.exists():
            raw = json.load(open(report_path))
            overall = raw.get("overall", raw)
            report = {k: overall.get(k) for k in ("RMSE", "MAE", "MAPE") if isinstance(overall, dict)}

        record = {
            "arm": arm, "round": r, "injected": injected,
            "exit_code": proc.returncode, "wall_seconds": round(wall, 1),
            "report": report,
            "trace": _latest_trace(),
            "recommended_edges": _graph_edge_metrics(graph_path),
        }
        if proc.returncode != 0:
            record["stderr_tail"] = proc.stderr[-2000:]
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{arm}] round {r}/{rounds}{' (inject)' if injected else ''} "
              f"exit={proc.returncode} wall={wall:.0f}s report={report}")
        if proc.returncode != 0:
            print(proc.stderr[-800:])
            raise SystemExit(f"[{arm}] round {r} 失败，实验中止")


def summarize() -> None:
    summary = {}
    for arm in ARMS:
        path = ARMS_DIR / arm / "rounds.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        selections = [tuple(r["trace"].get("final_selection") or []) for r in rows]
        changes = sum(1 for a, b in zip(selections, selections[1:]) if a != b)
        summary[arm] = {
            "rounds": len(rows),
            "rmse_series": [r["report"].get("RMSE") for r in rows],
            "mae_series": [r["report"].get("MAE") for r in rows],
            "wall_seconds": [r["wall_seconds"] for r in rows],
            "schedule_changes": changes,
            "selections": [list(s) for s in selections],
            "inject_round": INJECT_ROUND,
        }
    out = EXP_DIR / "summary.json"
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[ok] {out}")
    for arm, s in summary.items():
        print(arm, "| rounds:", s["rounds"], "| changes:", s["schedule_changes"],
              "| rmse:", [round(x) if x else None for x in s["rmse_series"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "run", "summarize", "restore"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        arms = ARMS if args.all else ([args.arm] if args.arm else None)
        if not arms:
            raise SystemExit("run 需要 --arm 或 --all")
        for arm in arms:
            run_arm(arm, args.rounds)
        summarize()
    elif args.action == "summarize":
        summarize()
    elif args.action == "restore":
        _restore_reports_from_baseline()
        _restore_data()
        print("[ok] reports/ 与数据已恢复到基线")


if __name__ == "__main__":
    main()
