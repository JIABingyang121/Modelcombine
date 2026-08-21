"""KG config and environment-driven constants."""

from __future__ import annotations

import os
from typing import List, Set

from src.eval.combination_utils import MODELS

def _env_str(key: str, default: str) -> str:
    return str(os.environ.get(key, default))

def _env_bool(key: str, default: bool) -> bool:
    default_str = "1" if default else "0"
    return _env_str(key, default_str).strip().lower() in {"1", "true", "yes", "on"}

def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except Exception:
        return int(default)

def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except Exception:
        return float(default)

def _env_str_set(key: str, default_csv: str) -> Set[str]:
    raw = _env_str(key, default_csv)
    items = [x.strip() for x in str(raw).split(",")]
    return {x for x in items if x}

PROTOCOL_B_VAL_FALLBACK_EPS = _env_float("MODELCOMBINE_KG_B_VAL_FALLBACK_EPS", 0.05)
PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION = _env_float("MODELCOMBINE_KG_B_POST_ADJUST_MAX_DEGRADATION", 0.05)
PROTOCOL_B_MIN_SELECTED_MODELS = _env_int("MODELCOMBINE_KG_B_MIN_SELECTED_MODELS", 3)
PROTOCOL_B_SMALL_SAMPLE_THRESHOLD = _env_int("MODELCOMBINE_KG_B_SMALL_SAMPLE_THRESHOLD", 500)
PROTOCOL_B_SMALL_SAMPLE_MIN_IMPROVEMENT = _env_float("MODELCOMBINE_KG_B_SMALL_SAMPLE_MIN_IMPROVEMENT", 0.01)
PROTOCOL_B_LONG_H_MIN_IMPROVEMENT = max(
    _env_float("MODELCOMBINE_KG_B_LONG_H_MIN_IMPROVEMENT", 0.001),
    0.0,
)
PROTOCOL_B_FEATURE_BONUS_WEIGHT = _env_float("MODELCOMBINE_KG_B_FEATURE_BONUS_WEIGHT", 0.8)
PROTOCOL_B_CORR_PENALTY_WEIGHT = _env_float("MODELCOMBINE_KG_B_CORR_PENALTY_WEIGHT", 0.2)
PROTOCOL_B_ADJUST_BONUS_SCALE = _env_float("MODELCOMBINE_KG_B_ADJUST_BONUS_SCALE", 0.8)
PROTOCOL_B_ADJUST_PENALTY_SCALE = _env_float("MODELCOMBINE_KG_B_ADJUST_PENALTY_SCALE", 0.2)
PROTOCOL_B_GLOBAL_MIN_IMPROVEMENT = max(
    _env_float("MODELCOMBINE_KG_B_GLOBAL_MIN_IMPROVEMENT", 0.0005),
    0.0,
)

PROTOCOL_B_REASONING_MODE = _env_str("MODELCOMBINE_KG_B_REASONING_MODE", "hybrid").strip().lower()
if PROTOCOL_B_REASONING_MODE not in {"off", "hybrid", "prefer"}:
    PROTOCOL_B_REASONING_MODE = "hybrid"

PROTOCOL_B_LARGE_SAMPLE_THRESHOLD = _env_int("MODELCOMBINE_KG_B_LARGE_SAMPLE_THRESHOLD", 5000)

PROTOCOL_A_SMALL_SAMPLE_LONG_H_MAX_VAL_SAMPLES = _env_int(
    "MODELCOMBINE_KG_A_SMALL_SAMPLE_LONG_H_MAX_VAL_SAMPLES", 500
)
PROTOCOL_A_SMALL_SAMPLE_LONG_H_MIN_H = _env_int("MODELCOMBINE_KG_A_SMALL_SAMPLE_LONG_H_MIN_H", 24)
PROTOCOL_A_SMALL_SAMPLE_LONG_H_TAIL_RATIO = min(
    max(_env_float("MODELCOMBINE_KG_A_SMALL_SAMPLE_LONG_H_TAIL_RATIO", 0.35), 0.1),
    0.8,
)
PROTOCOL_A_SMALL_SAMPLE_LONG_H_MIN_TAIL = _env_int("MODELCOMBINE_KG_A_SMALL_SAMPLE_LONG_H_MIN_TAIL", 40)
PROTOCOL_A_SMALL_SAMPLE_LONG_H_MAX_TAIL_DEGRADATION = _env_float(
    "MODELCOMBINE_KG_A_SMALL_SAMPLE_LONG_H_MAX_TAIL_DEGRADATION", 0.01
)

KG_INCLUDE_SEASONAL_NAIVE = _env_bool("MODELCOMBINE_KG_INCLUDE_SEASONAL_NAIVE", True)
KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT = _env_bool(
    "MODELCOMBINE_KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT",
    True,
)

DRIFT_PSI_LOW_THRESHOLD = _env_float("MODELCOMBINE_KG_DRIFT_PSI_LOW_THRESHOLD", 0.15)
DRIFT_PSI_HIGH_THRESHOLD = max(
    _env_float("MODELCOMBINE_KG_DRIFT_PSI_HIGH_THRESHOLD", 0.30),
    DRIFT_PSI_LOW_THRESHOLD,
)
# PSI deadband: avoid route flipping near threshold when drift is borderline.
# Default keeps high threshold consistent with drift high threshold (0.30),
# while preserving a small band below it for stabilization.
PROTOCOL_B_PSI_DEADBAND_HIGH = max(
    _env_float("MODELCOMBINE_KG_B_PSI_DEADBAND_HIGH", DRIFT_PSI_HIGH_THRESHOLD),
    DRIFT_PSI_LOW_THRESHOLD,
)
PROTOCOL_B_PSI_DEADBAND_LOW = min(
    max(
        _env_float(
            "MODELCOMBINE_KG_B_PSI_DEADBAND_LOW",
            max(DRIFT_PSI_LOW_THRESHOLD, PROTOCOL_B_PSI_DEADBAND_HIGH - 0.02),
        ),
        DRIFT_PSI_LOW_THRESHOLD,
    ),
    PROTOCOL_B_PSI_DEADBAND_HIGH,
)
DRIFT_HIGH_GUARD_FALLBACK_EPS = _env_float("MODELCOMBINE_KG_DRIFT_HIGH_GUARD_FALLBACK_EPS", 0.02)

KG_RIDGE_TEMPORAL_DECAY = max(_env_float("MODELCOMBINE_KG_RIDGE_TEMPORAL_DECAY", 0.003), 0.0)
KG_TEMPORAL_WEIGHT_MIN_RATIO = min(
    max(_env_float("MODELCOMBINE_KG_TEMPORAL_WEIGHT_MIN_RATIO", 0.05), 0.0),
    1.0,
)
KG_DRIFT_DECAY_MAX_MULTIPLIER = max(
    _env_float("MODELCOMBINE_KG_DRIFT_DECAY_MAX_MULTIPLIER", 5.0),
    1.0,
)
KG_GUARD_NAIVE_RELAX_RATIO = max(_env_float("MODELCOMBINE_KG_GUARD_NAIVE_RELAX_RATIO", 1.2), 1.0)

PROTOCOL_B_TAIL_HOLDOUT_RATIO = min(
    max(_env_float("MODELCOMBINE_KG_B_TAIL_HOLDOUT_RATIO", 0.30), 0.10),
    0.60,
)
PROTOCOL_B_TAIL_HOLDOUT_MAX_RATIO = max(_env_float("MODELCOMBINE_KG_B_TAIL_HOLDOUT_MAX_RATIO", 1.30), 1.0)
PROTOCOL_B_TAIL_HOLDOUT_MAX_RATIO_MEDIUM = max(
    _env_float("MODELCOMBINE_KG_B_TAIL_HOLDOUT_MAX_RATIO_MEDIUM", 1.22),
    1.0,
)
PROTOCOL_B_TAIL_HOLDOUT_MAX_RATIO_HIGH = max(
    _env_float("MODELCOMBINE_KG_B_TAIL_HOLDOUT_MAX_RATIO_HIGH", 1.15),
    1.0,
)

PROTOCOL_B_HIGH_DRIFT_VAL_FALLBACK_EPS = max(
    _env_float("MODELCOMBINE_KG_B_HIGH_DRIFT_VAL_FALLBACK_EPS", 0.01),
    0.0,
)
PROTOCOL_B_HIGH_DRIFT_OVERFIT_VAL_IMPROVE_THRESHOLD = max(
    _env_float("MODELCOMBINE_KG_B_HIGH_DRIFT_OVERFIT_VAL_IMPROVE_THRESHOLD", 0.05),
    0.0,
)
PROTOCOL_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_MAX = max(
    _env_float("MODELCOMBINE_KG_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_MAX", 1.02),
    1.0,
)
# P1: 短步长(h<=1)放宽 tail ratio，避免短期噪声触发过度回退
PROTOCOL_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_MAX_SHORT_H = max(
    _env_float("MODELCOMBINE_KG_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_MAX_SHORT_H", 1.08),
    1.0,
)
PROTOCOL_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_SHORT_H_MAX = _env_int(
    "MODELCOMBINE_KG_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_SHORT_H_MAX", 1
)
PROTOCOL_B_HIGH_DRIFT_A_PREFERRED_MIN_IMPROVE = max(
    _env_float("MODELCOMBINE_KG_B_HIGH_DRIFT_A_PREFERRED_MIN_IMPROVE", 0.05),
    0.0,
)
PROTOCOL_B_HIGH_DRIFT_A_PREFERRED_TAIL_RATIO = max(
    _env_float("MODELCOMBINE_KG_B_HIGH_DRIFT_A_PREFERRED_TAIL_RATIO", 1.10),
    1.0,
)
PROTOCOL_B_HIGH_DRIFT_SINGLE_MODEL_DIRECT_PASS_RATIO = max(
    _env_float("MODELCOMBINE_KG_B_HIGH_DRIFT_SINGLE_MODEL_DIRECT_PASS_RATIO", 1.01),
    1.0,
)
PROTOCOL_B_DISABLE_INTERACTION_HIGH_DRIFT_MIN_H = max(
    _env_int("MODELCOMBINE_KG_B_DISABLE_INTERACTION_HIGH_DRIFT_MIN_H", 6),
    1,
)
# NSW 专项：即使非 high drift，也在指定数据集上禁用 interaction residual
# 以降低复杂度，避免低信噪比下的交互项污染
PROTOCOL_B_DISABLE_INTERACTION_DATASETS = _env_str_set(
    "MODELCOMBINE_KG_B_DISABLE_INTERACTION_DATASETS",
    "",
)
PROTOCOL_B_LAST_BLOCK_GUARD_ENABLED = _env_bool(
    "MODELCOMBINE_KG_B_LAST_BLOCK_GUARD_ENABLED",
    True,
)
PROTOCOL_B_LAST_BLOCK_RATIO = min(
    max(_env_float("MODELCOMBINE_KG_B_LAST_BLOCK_RATIO", 0.20), 0.05),
    0.50,
)
PROTOCOL_B_LAST_BLOCK_MIN_SAMPLES = max(
    _env_int("MODELCOMBINE_KG_B_LAST_BLOCK_MIN_SAMPLES", 32),
    8,
)
PROTOCOL_B_HIGH_DRIFT_MIN_W_MULTIPLIER = max(
    _env_float("MODELCOMBINE_KG_B_HIGH_DRIFT_MIN_W_MULTIPLIER", 1.0),
    1.0,
)
PROTOCOL_B_COMPLEXITY_PENALTY_ENABLED = _env_bool(
    "MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_ENABLED",
    True,
)
PROTOCOL_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE = max(
    _env_float("MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE", 0.01),
    0.0,
)
# 复杂度惩罚的软边界比例：effective_threshold = min_rel_improve * soft_margin
# 例如 1% * 0.90 = 0.9%，避免 NSW 0.x% 的极小改进被硬阈值误杀。
PROTOCOL_B_COMPLEXITY_PENALTY_SOFT_MARGIN = min(
    max(_env_float("MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_SOFT_MARGIN", 0.90), 0.0),
    1.0,
)
# P1: 当 eligible 模型过少时降低 complexity 门槛（模型少→组合空间小→改进率天然低）
PROTOCOL_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE_FEW_MODELS = max(
    _env_float("MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE_FEW_MODELS", 0.003),
    0.0,
)
PROTOCOL_B_COMPLEXITY_PENALTY_FEW_MODELS_THRESHOLD = _env_int(
    "MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_FEW_MODELS_THRESHOLD", 5
)
PROTOCOL_B_COMPLEXITY_PENALTY_DATASETS = _env_str_set(
    "MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_DATASETS",
    "aemo_nsw",
)
PROTOCOL_B_COMPLEXITY_PENALTY_FALLBACK = _env_str(
    "MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_FALLBACK",
    "best_single",
).strip().lower()
if PROTOCOL_B_COMPLEXITY_PENALTY_FALLBACK not in {"best_single", "protocol_a"}:
    PROTOCOL_B_COMPLEXITY_PENALTY_FALLBACK = "best_single"
PROTOCOL_B_BEST_SINGLE_SCOPE = _env_str(
    "MODELCOMBINE_KG_B_BEST_SINGLE_SCOPE",
    "base_models_only",
).strip().lower()
if PROTOCOL_B_BEST_SINGLE_SCOPE not in {"all_eligible", "base_models_only"}:
    PROTOCOL_B_BEST_SINGLE_SCOPE = "base_models_only"
PROTOCOL_B_GUARD_BEST_SINGLE_SCOPE = _env_str(
    "MODELCOMBINE_KG_B_GUARD_BEST_SINGLE_SCOPE",
    PROTOCOL_B_BEST_SINGLE_SCOPE,
).strip().lower()
if PROTOCOL_B_GUARD_BEST_SINGLE_SCOPE not in {"all_eligible", "base_models_only"}:
    PROTOCOL_B_GUARD_BEST_SINGLE_SCOPE = PROTOCOL_B_BEST_SINGLE_SCOPE
PROTOCOL_B_ENFORCE_BASE_MODEL_IN_SELECTION = _env_bool(
    "MODELCOMBINE_KG_B_ENFORCE_BASE_MODEL_IN_SELECTION", True,
)
PROTOCOL_B_ENFORCE_BASE_MODEL_MIN_H = max(
    _env_int("MODELCOMBINE_KG_B_ENFORCE_BASE_MODEL_MIN_H", 6), 1,
)
PROTOCOL_B_ENFORCE_BASE_MODEL_DATASETS = _env_str_set(
    "MODELCOMBINE_KG_B_ENFORCE_BASE_MODEL_DATASETS",
    "",
)
PROTOCOL_B_ENFORCE_BASE_MODEL_MODE = _env_str(
    "MODELCOMBINE_KG_B_ENFORCE_BASE_MODEL_MODE",
    "inject_best_base",
).strip().lower()
if PROTOCOL_B_ENFORCE_BASE_MODEL_MODE not in {"inject_best_base", "best_base_single", "off"}:
    PROTOCOL_B_ENFORCE_BASE_MODEL_MODE = "inject_best_base"
PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_DATASETS = _env_str_set(
    "MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_DATASETS",
    "aemo_nsw",
)
PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_RATIO = min(
    max(_env_float("MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_RATIO", 0.20), 0.05),
    0.50,
)
PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_MIN_SAMPLES = max(
    _env_int("MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_MIN_SAMPLES", 48),
    8,
)
PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_MAX_DEGRADATION = max(
    _env_float("MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_MAX_DEGRADATION", 0.0),
    0.0,
)
PROTOCOL_B_MIN_W_OVERRIDE_DATASETS = _env_str_set(
    "MODELCOMBINE_KG_B_MIN_W_OVERRIDE_DATASETS",
    "aemo_nsw",
)
PROTOCOL_B_MIN_W_OVERRIDE_VALUE = min(
    max(_env_float("MODELCOMBINE_KG_B_MIN_W_OVERRIDE_VALUE", 0.12), 0.0),
    1.0,
)

PROTOCOL_STEPWISE_MIN_IMPROVEMENT = min(
    max(_env_float("MODELCOMBINE_KG_STEPWISE_MIN_IMPROVEMENT", 0.005), 0.0),
    0.05,
)
PROTOCOL_B_FEATURE_DIVERSITY_WEIGHT = max(
    _env_float("MODELCOMBINE_KG_B_FEATURE_DIVERSITY_WEIGHT", 0.25),
    0.0,
)

KG_STRICT_EXTENDED_POOL = _env_bool("MODELCOMBINE_KG_STRICT_EXTENDED_POOL", True)
KG_EXTENDED_POOL_MIN_LOADED = max(_env_int("MODELCOMBINE_KG_EXTENDED_POOL_MIN_LOADED", 1), 0)

MODEL_STABILITY_RMSE_NAIVE_MARGIN = max(_env_float("MODELCOMBINE_KG_RMSE_NAIVE_MARGIN", 1.05), 1.0)
# 长步长(h>=24)放宽 worse_than_naive 阈值：h24 模型方差大，1.05 过于激进会淘汰
# 太多候选导致 KG 退化为单模型。默认 1.20 允许 xgboost/arima 等保留。
MODEL_STABILITY_RMSE_NAIVE_MARGIN_LONG_H = max(
    _env_float("MODELCOMBINE_KG_RMSE_NAIVE_MARGIN_LONG_H", 1.20), 1.0,
)
# unstable_late 是否需要"准确度佐证"才能硬删除候选（合一计划 Task 6B）。
# 该规则只度量验证窗前后段误差比，不含任何准确度信息；实测中它把明显优于
# naive、且 validation 最优的候选直接删掉（见 result/ab_convergence/root_cause.md
# 与 same_round_quality.md：full 档 MAE 421.20 -> 恢复后 307.28）。
# 置真（默认）时，若 unstable_late 是**唯一**原因、且该候选通过既有准确度下限
# (MODEL_STABILITY_RMSE_NAIVE_MARGIN[_LONG_H])，则降级为软标记而不删除。
# 置假可复现修正前行为，用于 A/B 对照。注意：本开关同样影响 System B 实验路径。
MODEL_STABILITY_UNSTABLE_LATE_REQUIRES_CORROBORATION = _env_bool(
    "MODELCOMBINE_KG_UNSTABLE_LATE_REQUIRES_CORROBORATION", True
)

KG_HIGH_DRIFT_EXTENDED_SCORE_FACTOR = min(
    max(_env_float("MODELCOMBINE_KG_HIGH_DRIFT_EXTENDED_SCORE_FACTOR", 0.80), 0.1),
    1.0,
)
KG_ZERO_WEIGHT_CLEANUP_THRESHOLD = max(_env_float("MODELCOMBINE_KG_ZERO_WEIGHT_CLEANUP_THRESHOLD", 1e-4), 0.0)

KG_INCLUDE_RL_QMS = _env_bool("MODELCOMBINE_KG_INCLUDE_RL_QMS", True)
KG_INCLUDE_DASH_TTA = _env_bool("MODELCOMBINE_KG_INCLUDE_DASH_TTA", False)
KG_INCLUDE_BASELINE_STACKING = _env_bool("MODELCOMBINE_KG_INCLUDE_BASELINE_STACKING", True)
KG_INCLUDE_DYNAMIC_STACKING = _env_bool("MODELCOMBINE_KG_INCLUDE_DYNAMIC_STACKING", False)
KG_INCLUDE_STATIC_WEIGHT = _env_bool("MODELCOMBINE_KG_INCLUDE_STATIC_WEIGHT", False)
KG_INCLUDE_MOLE_ROUTER = _env_bool("MODELCOMBINE_KG_INCLUDE_MOLE_ROUTER", False)

HIGH_DRIFT_ALPHA_CANDIDATES = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

# P0-3: h24 反过拟合配置
# 长步长 Ridge alpha 倍率：h24 用更强正则化约束权重
KG_LONG_HORIZON_ALPHA_MULTIPLIER = max(
    _env_float("MODELCOMBINE_KG_LONG_HORIZON_ALPHA_MULTIPLIER", 3.0),
    1.0,
)
KG_LONG_HORIZON_MIN_H = _env_int("MODELCOMBINE_KG_LONG_HORIZON_MIN_H", 24)
# h24 max_models 硬上限：防止候选过多引入过拟合维度
KG_LONG_HORIZON_MAX_MODELS_CAP = _env_int("MODELCOMBINE_KG_LONG_HORIZON_MAX_MODELS_CAP", 3)
# Protocol B 多窗口一致性守卫（h>=24 时启用，3 窗口中 >=2 窗口恶化即回退）
PROTOCOL_B_MULTI_WINDOW_GUARD_ENABLED = _env_bool(
    "MODELCOMBINE_KG_B_MULTI_WINDOW_GUARD_ENABLED", True,
)
PROTOCOL_B_MULTI_WINDOW_GUARD_MIN_H = _env_int(
    "MODELCOMBINE_KG_B_MULTI_WINDOW_GUARD_MIN_H", 24,
)
PROTOCOL_B_MULTI_WINDOW_GUARD_MAX_DEGRADATION = max(
    _env_float("MODELCOMBINE_KG_B_MULTI_WINDOW_GUARD_MAX_DEGRADATION", 0.005),
    0.0,
)
PROTOCOL_B_MULTI_WINDOW_GUARD_MAJORITY = _env_int(
    "MODELCOMBINE_KG_B_MULTI_WINDOW_GUARD_MAJORITY", 2,
)

def _build_kg_model_candidates() -> List[str]:
    # seasonal_naive 作为扩展候选池节点处理，避免其与基础模型预测在
    # DST 重复时间戳场景下触发严格行对齐失败。
    base = [m for m in MODELS if m != "seasonal_naive"]
    # 支持通过环境变量动态注入额外基础候选（如 itransformer）
    # 格式：MODELCOMBINE_KG_EXTRA_BASE_CANDIDATES="itransformer,another_model"
    extra_raw = _env_str("MODELCOMBINE_KG_EXTRA_BASE_CANDIDATES", "").strip()
    if extra_raw:
        for m in extra_raw.split(","):
            m = m.strip()
            if m and m not in base:
                base.append(m)
    return base

def _build_extended_pool_strategies() -> List[str]:
    strategies = [
        "gating_network",
        "soft_gating",
        "scenario_bucket",
        "gating_network_v2",
        "adaptive_bucket",
        "scenario_similarity",
    ]
    if KG_INCLUDE_SEASONAL_NAIVE:
        strategies.append("seasonal_naive")
    if KG_INCLUDE_RL_QMS:
        strategies.append("rl_qms")
    if KG_INCLUDE_DASH_TTA:
        strategies.append("dash_tta")
    if KG_INCLUDE_BASELINE_STACKING:
        strategies.append("stacking_safe")
    if KG_INCLUDE_DYNAMIC_STACKING:
        strategies.append("dynamic_stacking")
    if KG_INCLUDE_STATIC_WEIGHT:
        strategies.append("static_weight_safe")
    if KG_INCLUDE_MOLE_ROUTER:
        strategies.append("mole_router")
    return strategies

def _extended_pool_penalty_models() -> set:
    """高漂移降权名单：仅对 KG component 动态策略降权，避免误伤外部强基线。"""
    return {
        "gating_network",
        "soft_gating",
        "scenario_bucket",
        "gating_network_v2",
        "adaptive_bucket",
        "scenario_similarity",
    }

# --- 运行时预测输出（System A/B 合一 Task 3.1）-------------------------------
# Protocol A/B 可选地把真实 pred_val / pred_test 交给调用方。默认关闭：实验脚本
# 的 JSON 结构与体积保持不变。开启后预测以 numpy 数组挂在该键下，**不可**写入
# SelectionTrace 或普通实验 JSON，调用方取用后应立即摘除。
RUNTIME_PREDICTIONS_KEY = "_runtime_predictions"
