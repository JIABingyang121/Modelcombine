from typing import Dict, Optional


# 统一策略命名层：
# - 保留原 strategy_id（用于兼容历史结果与脚本）
# - 新增 display_name / method_family / method_route / core_route

_STRATEGY_META: Dict[str, Dict[str, str]] = {
    # baseline classic
    "simple_avg_safe": {
        "display_name": "Baseline Classic / Simple Average (Safe)",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "simple_avg": {
        "display_name": "Baseline Classic / Simple Average",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "smart_weighted": {
        "display_name": "Baseline Classic / Smart Weighted",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "static_weight_safe": {
        "display_name": "Baseline Classic / Static Weight (Safe)",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "static_weight": {
        "display_name": "Baseline Classic / Static Weight",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "stacking_safe": {
        "display_name": "Baseline Classic / Stacking (Safe)",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "stacking": {
        "display_name": "Baseline Classic / Stacking",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "dynamic_avg": {
        "display_name": "Baseline Classic / Dynamic Avg",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "dynamic_weight": {
        "display_name": "Baseline Classic / Dynamic Weight",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "dynamic_stacking": {
        "display_name": "Baseline Classic / Dynamic Stacking",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    "constrained_opt": {
        "display_name": "Baseline Classic / Constrained Optimization",
        "method_family": "baseline_classic",
        "method_route": "baseline",
        "core_route": "no",
    },
    # baseline sota
    "rl_qms": {
        "display_name": "Baseline SOTA / RL-QMS",
        "method_family": "baseline_sota",
        "method_route": "baseline",
        "core_route": "no",
    },
    "mole_router": {
        "display_name": "Baseline SOTA / MoLE Router",
        "method_family": "baseline_sota",
        "method_route": "baseline",
        "core_route": "no",
    },
    "dash_tta": {
        "display_name": "KG Node / DASH-TTA",
        "method_family": "kg_node",
        "method_route": "kg_node",
        "core_route": "no",
    },
    # KG component branches (non-core)
    "gating_network": {
        "display_name": "KG Component / NN Gating v1",
        "method_family": "kg_component_nn",
        "method_route": "ablation",
        "core_route": "no",
    },
    "soft_gating": {
        "display_name": "KG Component / Soft Gating",
        "method_family": "kg_component_nn",
        "method_route": "ablation",
        "core_route": "no",
    },
    "gating_network_v2": {
        "display_name": "KG Component / NN Gating v2",
        "method_family": "kg_component_nn",
        "method_route": "ablation",
        "core_route": "no",
    },
    "scenario_bucket": {
        "display_name": "KG Component / Scenario Bucket",
        "method_family": "kg_component_scenario",
        "method_route": "ablation",
        "core_route": "no",
    },
    "adaptive_bucket": {
        "display_name": "KG Component / Adaptive Bucket",
        "method_family": "kg_component_scenario",
        "method_route": "ablation",
        "core_route": "no",
    },
    "scenario_similarity": {
        "display_name": "KG Component / Scenario Similarity",
        "method_family": "kg_component_scenario",
        "method_route": "ablation",
        "core_route": "no",
    },
    # KG core route
    "kg_protocol_a": {
        "display_name": "KG Core / Protocol A (Pred-Only)",
        "method_family": "kg_core",
        "method_route": "core_kg",
        "core_route": "yes",
    },
    "kg_protocol_b": {
        "display_name": "KG Core / Protocol B (Pred+Feature)",
        "method_family": "kg_core",
        "method_route": "core_kg",
        "core_route": "yes",
    },
}

_ALIASES = {
    "protocol_A": "kg_protocol_a",
    "protocol_B": "kg_protocol_b",
    "protocol_a": "kg_protocol_a",
    "protocol_b": "kg_protocol_b",
}


def normalize_strategy_id(strategy_id: str) -> str:
    if strategy_id in _ALIASES:
        return _ALIASES[strategy_id]
    return strategy_id


def get_strategy_naming(strategy_id: str, default_category: Optional[str] = None) -> Dict[str, str]:
    normalized = normalize_strategy_id(strategy_id)
    meta = _STRATEGY_META.get(normalized)
    if meta is None:
        # 回退规则：尽量保留可读性，不阻断主流程
        return {
            "strategy_id": normalized,
            "display_name": normalized,
            "method_family": default_category or "unknown",
            "method_route": "unknown",
            "core_route": "unknown",
        }
    return {
        "strategy_id": normalized,
        "display_name": meta["display_name"],
        "method_family": meta["method_family"],
        "method_route": meta["method_route"],
        "core_route": meta["core_route"],
    }
