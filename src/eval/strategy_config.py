import os

# 组合策略相关配置常量（从 data_utils 中拆分，避免语义混杂）

# SoftGating alpha 候选值
SOFTGATING_ALPHA_CANDIDATES = [0.1, 0.3, 0.5, 0.7, 0.9]

# kg_component 策略模型池模式：
# - safe: 与 baseline(static/stacking) 对齐，使用 safe_cols
# - scene_fit: 使用 scene_fit_cols（场景受限后）
KG_COMPONENT_MODEL_POOL_MODE = os.environ.get("MODELCOMBINE_KG_COMPONENT_POOL_MODE", "safe").strip().lower()
if KG_COMPONENT_MODEL_POOL_MODE not in {"safe", "scene_fit"}:
    KG_COMPONENT_MODEL_POOL_MODE = "safe"

# DirectWeightGating 温度
try:
    DIRECT_WEIGHT_GATING_TEMPERATURE = float(
        os.environ.get("MODELCOMBINE_DIRECT_WEIGHT_TEMPERATURE", "0.5")
    )
except Exception:
    DIRECT_WEIGHT_GATING_TEMPERATURE = 0.5

# ScenarioSimilarity 权重模式：
# - error_softmax: 基于误差幅度的 softmax（默认）
# - rank: 基于排名的稳健权重
# - softmax: 原始 softmax(-error / T)
SCENARIO_SIMILARITY_WEIGHT_MODE = os.environ.get(
    "MODELCOMBINE_SCENARIO_SIM_WEIGHT_MODE", "error_softmax"
).strip().lower()
if SCENARIO_SIMILARITY_WEIGHT_MODE not in {"error_softmax", "rank", "softmax"}:
    SCENARIO_SIMILARITY_WEIGHT_MODE = "error_softmax"

try:
    SCENARIO_SIMILARITY_TEMPERATURE = float(
        os.environ.get("MODELCOMBINE_SCENARIO_SIM_TEMPERATURE", "2.0")
    )
except Exception:
    SCENARIO_SIMILARITY_TEMPERATURE = 2.0

# 漂移感知时间衰减参数：
# beta = min(max, base + slope * max_psi)
try:
    DRIFT_DECAY_BASE = float(os.environ.get("MODELCOMBINE_DRIFT_DECAY_BASE", "0.001"))
except Exception:
    DRIFT_DECAY_BASE = 0.001
try:
    DRIFT_DECAY_SLOPE = float(os.environ.get("MODELCOMBINE_DRIFT_DECAY_SLOPE", "0.005"))
except Exception:
    DRIFT_DECAY_SLOPE = 0.005
try:
    DRIFT_DECAY_MAX = float(os.environ.get("MODELCOMBINE_DRIFT_DECAY_MAX", "0.05"))
except Exception:
    DRIFT_DECAY_MAX = 0.05
