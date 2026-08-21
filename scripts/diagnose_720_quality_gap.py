"""720 档剩余质量差距的预测阶梯诊断（合一计划 Task 6C，只诊断不修改）。

**背景**：Task 6B 修正 `unstable_late` 后，full 档门槛通过，720 档完全未变：
Protocol B 最终 MAE 348.5613，而 validation 选出的最佳单模型 343.0834，
两者选中的都是 `multimodal_fusion`，差 5.4779（1.60%）。Protocol B 的权重为
0.9993846，但 `linear_reconstruction_match=false`——说明差距发生在**模型选定
之后的预测变换环节**，而不是候选排序或 `unstable_late`。

**本脚本只做测量与单变量对照，不改任何阈值、guard 或候选池默认值。**
所有配置均在同一进程内临时改写并在 finally 中还原；候选预测矩阵**只构建一次**，
全部对照共用，杜绝重训带来的差异。

预测阶梯：
    L0 原始 multimodal_fusion
    L1 Ridge 权重线性重建（w * mf）
    L2 引擎最终输出（含 interaction / post_adjustment 等一切后处理）

单变量对照：
    - 禁用 interaction（经 PROTOCOL_B_DISABLE_INTERACTION_DATASETS）
    - 禁用 post_adjustment（经把降级容忍度设为 -1，使 sanity check 必然拒绝）
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import scripts.compare_system_ab_same_round as same_round  # noqa: E402
import src.eval.kg.protocol_b as pb  # noqa: E402
from src.eval.kg.config import RUNTIME_PREDICTIONS_KEY  # noqa: E402
from src.pipeline.prediction_pool import build_region_prediction_bundle  # noqa: E402

ROWS = "720"


def mae(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m])))


@contextmanager
def temporarily(**overrides: Any):
    """在 pb 模块命名空间内临时改写配置，finally 必定还原。"""
    saved = {k: getattr(pb, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(pb, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(pb, k, v)


def run_engine(bundle, region: str) -> Dict[str, Any]:
    """在既有预测矩阵上跑一次 Protocol B，返回最终预测与完整判据。"""
    raw = pb.kg_combination_with_features(
        bundle.df_val,
        bundle.df_test,
        bundle.df_raw_val,
        bundle.df_raw_test,
        list(bundle.model_cols),
        1,
        dataset_name=region,
        base_model_cols=list(bundle.base_model_cols),
        return_predictions=True,
    )
    preds = raw.get(RUNTIME_PREDICTIONS_KEY) or {}
    split = raw.get("test") or {}
    wm = split.get("weight_meta") or {}
    return {
        "protocol": raw.get("protocol"),
        "selected_models": split.get("selected_models"),
        "weights": split.get("weights"),
        "pred_val": preds.get("val"),
        "pred_test": preds.get("test"),
        "interaction_branch": wm.get("interaction_branch"),
        "post_adjustment": wm.get("post_adjustment"),
        "reported_test_mae": split.get("mae"),
        "reported_val_mae": (raw.get("val") or {}).get("mae"),
    }


def ladder(bundle, res: Dict[str, Any], best_single: str) -> List[Dict[str, Any]]:
    y_val = np.asarray(bundle.df_val["y"].values, dtype=float)
    y_test = np.asarray(bundle.df_test["y"].values, dtype=float)
    mf_val = np.asarray(bundle.df_val[best_single].values, dtype=float)
    mf_test = np.asarray(bundle.df_test[best_single].values, dtype=float)

    sel = list(res["selected_models"] or [])
    w = dict(res["weights"] or {})
    lin_val = bundle.df_val[sel].to_numpy(float) @ np.array([w[m] for m in sel])
    lin_test = bundle.df_test[sel].to_numpy(float) @ np.array([w[m] for m in sel])

    levels = [
        ("L0_raw_best_single", mf_val, mf_test),
        ("L1_ridge_linear_reconstruction", lin_val, lin_test),
        ("L2_engine_final", np.asarray(res["pred_val"], float), np.asarray(res["pred_test"], float)),
    ]
    out: List[Dict[str, Any]] = []
    prev_test: Optional[np.ndarray] = None
    for name, pv, pt in levels:
        row = {
            "level": name,
            "val_mae": round(mae(y_val, pv), 6),
            "test_mae": round(mae(y_test, pt), 6),
        }
        if prev_test is not None:
            d = pt - prev_test
            row["test_delta_vs_prev"] = {
                "max_abs": round(float(np.max(np.abs(d))), 6),
                "mean_abs": round(float(np.mean(np.abs(d))), 6),
                "mean_signed": round(float(np.mean(d)), 6),
            }
            row["test_mae_delta_vs_prev"] = round(row["test_mae"] - out[-1]["test_mae"], 6)
        prev_test = pt
        out.append(row)
    return out


def main() -> int:
    train, test, region, split_meta = same_round._load_and_split(ROWS)
    validation_days = same_round._validation_days(train, ROWS)

    # 候选预测矩阵只构建一次，所有对照共用（不重训）
    bundle = build_region_prediction_bundle(
        region=region,
        train=train,
        test=test,
        candidate_models=[
            m for m in __import__("src.models.registry", fromlist=["x"]).model_registry
            .get_available_models() if "blender" not in m.lower()
        ],
        validation_days=validation_days,
    )
    best = same_round.best_single_summaries(bundle)
    bs_name = (best.get("validation_selected") or {}).get("model")

    baseline = run_engine(bundle, region)
    report: Dict[str, Any] = {
        "rows": ROWS,
        "region": region,
        "split": split_meta,
        "validation_days": validation_days,
        "model_cols": list(bundle.model_cols),
        "best_single_validation_selected": best.get("validation_selected"),
        "baseline": {
            k: v for k, v in baseline.items() if k not in ("pred_val", "pred_test")
        },
        "prediction_ladder": ladder(bundle, baseline, bs_name),
    }

    # --- 单变量对照：共用同一 bundle，不重训 ---
    variants: Dict[str, Dict[str, Any]] = {}

    disabled = set(getattr(pb, "PROTOCOL_B_DISABLE_INTERACTION_DATASETS", set())) | {region}
    with temporarily(PROTOCOL_B_DISABLE_INTERACTION_DATASETS=disabled):
        r = run_engine(bundle, region)
        variants["interaction_disabled"] = {
            **{k: v for k, v in r.items() if k not in ("pred_val", "pred_test")},
            "ladder": ladder(bundle, r, bs_name),
        }

    # ⚠️ 该常量并非 post_adjustment 专用：protocol_b.py 的 589/598/613/626 行
    # （interaction 的 val/tail/cv 接受判据）与 722 行（post_adjustment）共用它。
    # 因此下面这一组**不是单变量**，仅作留档，不能用于归因 post_adjustment。
    # post_adjustment 是否运行改以其自身元数据为准（单模型时该分支根本不进入）。
    with temporarily(PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION=-1.0):
        r = run_engine(bundle, region)
        variants["CONFOUNDED_shared_degradation_constant"] = {
            **{k: v for k, v in r.items() if k not in ("pred_val", "pred_test")},
            "warning": (
                "not single-variable: PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION also gates "
                "interaction acceptance (protocol_b.py:589,598,613,626)"
            ),
            "ladder": ladder(bundle, r, bs_name),
        }

    # 泛化性探针：同一预测矩阵下，把 test 切成若干子窗，比较 interaction 开/关。
    # 若 interaction 在多数子窗都更差 -> 系统性；若只由个别子窗驱动 -> 更像波动。
    with temporarily(PROTOCOL_B_DISABLE_INTERACTION_DATASETS=disabled):
        off = run_engine(bundle, region)
    y_test = np.asarray(bundle.df_test["y"].values, dtype=float)
    on_pred = np.asarray(baseline["pred_test"], dtype=float)
    off_pred = np.asarray(off["pred_test"], dtype=float)
    n_win = 6
    edges = np.linspace(0, len(y_test), n_win + 1, dtype=int)
    sub = []
    for i in range(n_win):
        a, b = edges[i], edges[i + 1]
        if b - a < 2:
            continue
        m_on, m_off = mae(y_test[a:b], on_pred[a:b]), mae(y_test[a:b], off_pred[a:b])
        sub.append({
            "window": f"[{a},{b})",
            "n": int(b - a),
            "mae_interaction_on": round(m_on, 6),
            "mae_interaction_off": round(m_off, 6),
            "interaction_worse": bool(m_on > m_off),
        })
    report["generalization_probe"] = {
        "note": "同一预测矩阵、同一 test 标签；仅切分子窗对比 interaction 开/关",
        "n_windows_interaction_worse": sum(1 for w in sub if w["interaction_worse"]),
        "n_windows": len(sub),
        "windows": sub,
    }

    report["variants"] = variants
    report["config_restored"] = {
        "PROTOCOL_B_DISABLE_INTERACTION_DATASETS": sorted(
            getattr(pb, "PROTOCOL_B_DISABLE_INTERACTION_DATASETS", set())
        ),
        "PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION": pb.PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION,
    }

    out = Path(PROJECT_ROOT) / "result" / "ab_convergence" / "diagnose_720_gap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[720-diag] -> {out} ({out.stat().st_size/1024:.1f} KB)")
    print(json.dumps(report["prediction_ladder"], indent=2, ensure_ascii=False))
    for name, v in variants.items():
        print(f"\n[{name}] protocol={v['protocol']} test_mae={v['reported_test_mae']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
