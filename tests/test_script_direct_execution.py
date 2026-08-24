"""直接执行（python scripts/*.py）时脚本必须自行把仓库根加入 sys.path。

pytest 会替脚本把仓库根放进 sys.path，因此这类 bug 只有通过子进程直接执行才能暴露。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_validate_protocol_b_v6_resolves_scripts_when_run_directly(tmp_path):
    """脚本顶部 `from scripts.* import ...` 必须在直接执行时可用（--help 即触发）。"""
    proc = _run("scripts/validate_protocol_b_v6.py", "--help", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "No module named 'scripts'" not in proc.stderr


def test_candidate_diagnostic_resolves_scripts_when_run_directly(tmp_path):
    """诊断脚本在 main() 内的 `from scripts.* import ...` 必须在直接执行时可用。

    用假路径跑到来源校验，应失败在缺文件/来源，而不是导入错误。
    """
    proc = _run(
        "scripts/run_protocol_b_candidate_diagnostic.py",
        "--pred-root", str(tmp_path),
        "--raw-root", str(tmp_path),
        "--feature-root", str(tmp_path),
        "--pipeline-config", str(tmp_path / "pipeline.yaml"),
        "--output", str(tmp_path / "out.json"),
        "--out-root", str(tmp_path / "traces"),
        cwd=tmp_path,
    )
    assert "No module named 'scripts'" not in proc.stderr
    assert proc.returncode != 0  # 会因缺数据失败，但不应是导入错误
