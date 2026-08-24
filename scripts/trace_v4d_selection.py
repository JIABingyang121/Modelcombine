"""追溯 v4d（commit 63d2def）的 VIC h=24 决策链来源（Task 8.3 Task 6 阶段 D）。

诊断专用：在 detached v4d worktree 上只包装函数调用并记录各阶段输入/输出，
**不改返回值**。结果只用于解释历史 `[catboost_reg, lgbm_reg]` 的来源，不改变
已写定的 v6 门槛。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

LEGACY_EXPECTED_HEAD = "63d2defe4889bc82cded69036fc9ca8987192b19"


def _git_head(legacy_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(legacy_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot read git HEAD of legacy root: {result.stderr.strip()}")
    return result.stdout.strip()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def wrap_call(log: List[Dict[str, Any]], name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """包装函数并记录结果，不修改返回值。"""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        log.append({"stage": name, "result": json_safe(result)})
        return result

    wrapped.__name__ = name
    return wrapped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--dataset", default="aemo_vic")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # 校验 legacy 目录确实是 v4d（63d2def），传错目录立即失败。
    head = _git_head(args.legacy_root)
    if head != LEGACY_EXPECTED_HEAD:
        raise RuntimeError(
            f"legacy root is not v4d 63d2def: HEAD={head}; expected {LEGACY_EXPECTED_HEAD}"
        )

    # 校验与 v6 同一份锁定来源（同一 baselines_v5 + 同一 pipeline 配置）。
    from scripts.run_protocol_b_candidate_diagnostic import (
        LOCKED_DATA_SHA256,
        LOCKED_PIPELINE_SHA256,
        verify_sha256,
    )

    verify_sha256(args.pipeline_config, LOCKED_PIPELINE_SHA256, label="pipeline config")
    verify_sha256(
        args.feature_root / args.dataset / "train.csv",
        LOCKED_DATA_SHA256[args.dataset],
        label=f"{args.dataset} train.csv",
    )

    # 先隔离：任何当前 worktree 的项目模块都不允许已加载，否则包装的是当前代码。
    for mod in ("src", "scripts", "scripts.run_system_ab_shadow", "scripts.train_baselines"):
        if mod in sys.modules:
            raise RuntimeError(
                f"current-worktree module already imported: {mod}; "
                "trace_v4d_selection must run in a clean process against the detached v4d checkout"
            )

    legacy_root = str(args.legacy_root.resolve())
    sys.path.insert(0, legacy_root)

    # 仅在此刻才导入 v4d 模块。
    import scripts.run_system_ab_shadow as v4d_shadow  # noqa: E402
    import src.eval.kg.conflict as v4d_conflict  # noqa: E402
    import src.eval.kg.model_selection as v4d_ms  # noqa: E402

    log: List[Dict[str, Any]] = []
    for name in ("select_models_protocol_b", "filter_conflicting_models", "_fallback_selection", "fit_static_weight_ridge"):
        target = getattr(v4d_ms, name, None)
        if target is None:
            target = getattr(v4d_conflict, name, None)
        if target is None:
            log.append({"stage": name, "result": None, "missing": True})
            continue
        wrapped = wrap_call(log, name, target)
        if hasattr(v4d_ms, name):
            setattr(v4d_ms, name, wrapped)
        else:
            setattr(v4d_conflict, name, wrapped)

    # 复用 v4d 的 build_task_matrix 与 Protocol B 运行，只记录、不改返回值。
    matrix = v4d_shadow.build_task_matrix(
        dataset=args.dataset,
        horizon=args.horizon,
        models=v4d_shadow._build_kg_model_candidates(),
        pred_root=args.pred_root,
        raw_root=args.raw_root,
    )
    from src.eval.kg.protocol_b import kg_combination_with_features  # noqa: E402

    raw = kg_combination_with_features(
        matrix["df_val_kg"], matrix["df_test_kg"],
        matrix["df_raw_val"], matrix["df_raw_test"],
        matrix["safe_models"], args.horizon,
        dataset_name=args.dataset, base_model_cols=matrix["base_model_cols"],
    )

    selection_flow = None
    val = raw.get("val") or {}
    wm = val.get("weight_meta") if isinstance(val, dict) else {}
    if isinstance(wm, dict):
        sm = wm.get("protocol_b_selection_meta") or {}
        selection_flow = sm.get("selection_flow") if isinstance(sm, dict) else None

    report = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "legacy_root": legacy_root,
        "legacy_head": head,
        "wrapped_calls": log,
        "final_models": list((val or {}).get("selected_models") or []),
        "protocol": raw.get("protocol"),
        "selection_flow": json_safe(selection_flow),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v4d selection trace written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
