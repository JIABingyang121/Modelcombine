import subprocess
import sys
from src.core.scenario_id import compute_scenario_id


def test_deterministic_same_input():
    fields = {"mean_load": 100.5, "cv_load": 0.2, "region_type": 1.0}
    assert compute_scenario_id(fields) == compute_scenario_id(fields)


def test_order_independent():
    a = compute_scenario_id({"x": 1.0, "y": 2.0})
    b = compute_scenario_id({"y": 2.0, "x": 1.0})
    assert a == b


def test_prefix_preserved_for_fuzzy_match():
    # main.py 用 `region in sid` 做模糊匹配，前缀必须保留
    sid = compute_scenario_id({"mean_load": 1.0}, prefix="PJME")
    assert sid.startswith("PJME_")
    assert "PJME" in sid


def test_different_input_different_id():
    assert compute_scenario_id({"a": 1.0}) != compute_scenario_id({"a": 2.0})


def test_stable_across_processes():
    # 关键回归：跨独立进程（模拟跨运行）必须产出相同 id，
    # 这正是 Python 内置 hash() 做不到、当前 bug 的根因。
    import os
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent  # tests/ 的上一级 = 仓库根
    code = (
        "from src.core.scenario_id import compute_scenario_id;"
        "print(compute_scenario_id({'mean_load': 42.0, 'cv': 0.1}, prefix='R'))"
    )
    # 显式传 cwd + PYTHONPATH，避免依赖调用方 cwd
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    r1 = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, cwd=str(repo_root), env=env)
    r2 = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, cwd=str(repo_root), env=env)
    assert r1.returncode == 0, f"subprocess failed: {r1.stderr}"
    assert r1.stdout.strip() == r2.stdout.strip()
    assert r1.stdout.strip() != ""
