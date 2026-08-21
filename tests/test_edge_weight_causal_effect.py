"""最小因果验证实验：图谱关系强度（边权）是否真的影响最终模型组合选择。

背景（见 项目总纲大纲.md §6.2、§9、§10 P1#2）：固定边权/移动平均/Hawkes 三策略
27 轮正式对照实验显示三组最终选择和误差完全一致，但混有历史场景库跨轮累积等
混杂因素，无法单独归因到"边权是否被消费"。

本测试不跑多轮 pipeline，而是直接对 PowerModelCombinator.select_optimal_path
做单次、确定性调用：构造两条候选路径，只翻转 scenario->path 的 recommended_for
边权（0.95/0.05 vs 0.05/0.95），每次用全新的 ModelGraph + combinator 实例，
彻底排除历史场景库/图谱状态跨轮残留的混杂因素。

诊断性测试，锁定"当前代码下边权变化是否改变最终选择"这一现状证据，不预设
应该怎么修。若断言开始失败，说明有人改动了消费链路，需要重新评估 §6.2/§10
的结论是否仍然成立。

**实际结果比最初假设更细致（写这份测试之前先入为主猜的是"完全没有任何路径
能感知边权"，跑出来发现是错的，已按实测结果改写断言，过程见对话记录）**：
边权对最终选择*并非*彻底零影响——`_evaluate_path_cost` 的 error 估计确实不读
边权，但 `_generate_candidate_paths` 按图谱推理得分（读边权）排序后把候选路径
依次 append 进候选列表，而 SLA 优化器 `ParetoOptimizer.select_from_pareto` 在
多个候选 Pareto 分数**完全打平**时用 `np.argmin`，会取列表中第一个——也就是
说边权能通过"候选顺序"间接左右一个精确平局的 tie-break。但现实历史数据里
两条路径的 RMSE 几乎不可能位级精确相等；只要存在任何非零的真实误差差异
（哪怕 0.5%），`_evaluate_path_cost` 算出的 error 差异就完全压过 tie-break，
边权无论取多极端的值都不再改变最终选择。这精确解释了 27 轮真实实验里
"边权确实变了、但最终选择和误差完全一致"的现象：不是因为边权信号在哪里
丢失了，是因为它只在一个真实数据中几乎不会出现的"精确平局"退化情形下才
生效。
"""
from src.graph.model_graph import ModelGraph
from src.selector.combinator import PowerModelCombinator

SCENARIO_ID = "causal_test_scenario"
FEATURE_ID = "feat_x"
MODEL_A = "model_a"
MODEL_B = "model_b"
PATH_A = f"single_{MODEL_A}"
PATH_B = f"single_{MODEL_B}"


def _build_graph(weight_a: float, weight_b: float) -> ModelGraph:
    """构造一张图：两条候选路径，只有 scenario->path 边权不同。"""
    mg = ModelGraph()
    mg.add_scenario_node(SCENARIO_ID, {})
    mg.add_feature_node(FEATURE_ID)
    mg.add_model_node(MODEL_A, {"input_constraints": {"features": [FEATURE_ID]}})
    mg.add_model_node(MODEL_B, {"input_constraints": {"features": [FEATURE_ID]}})
    mg.add_feature_model_edge(FEATURE_ID, MODEL_A)
    mg.add_feature_model_edge(FEATURE_ID, MODEL_B)

    mg.instantiate_path(PATH_A, [MODEL_A], strategy="single")
    mg.instantiate_path(PATH_B, [MODEL_B], strategy="single")
    mg.add_scenario_path_edge(SCENARIO_ID, PATH_A, performance_score=weight_a)
    mg.add_scenario_path_edge(SCENARIO_ID, PATH_B, performance_score=weight_b)
    return mg


def _build_combinator(rmse_a: float, rmse_b: float) -> PowerModelCombinator:
    combinator = PowerModelCombinator(
        enable_adaptive_weights=False,
        enable_resource_prediction=False,
    )
    combinator.set_historical_scenarios(
        [
            ("hist_a", {}, {"path_id": PATH_A, "path_rmse": rmse_a}),
            ("hist_b", {}, {"path_id": PATH_B, "path_rmse": rmse_b}),
        ]
    )
    return combinator


def _select(weight_a: float, weight_b: float, rmse_a: float, rmse_b: float) -> dict:
    mg = _build_graph(weight_a, weight_b)
    combinator = _build_combinator(rmse_a, rmse_b)
    return combinator.select_optimal_path(
        scenario_signature={"_scenario_id": SCENARIO_ID},
        available_models=[MODEL_A, MODEL_B],
        constraints={},
        model_graph=mg,
        similar_scenarios=[("hist_a", 1.0), ("hist_b", 1.0)],
        actual_data_columns={FEATURE_ID},
    )


def test_graph_reasoning_ranking_tracks_edge_weight_positive_control():
    """正控制：图谱推理层本身确实读边权并跟着变——证明信号没有在图谱层丢失。"""
    mg_a_favored = _build_graph(weight_a=0.95, weight_b=0.05)
    top_when_a_favored = dict(
        mg_a_favored.infer_optimal_path_by_reasoning(
            SCENARIO_ID, {FEATURE_ID}, constraints={"max_models": 3}
        )
    )

    mg_b_favored = _build_graph(weight_a=0.05, weight_b=0.95)
    top_when_b_favored = dict(
        mg_b_favored.infer_optimal_path_by_reasoning(
            SCENARIO_ID, {FEATURE_ID}, constraints={"max_models": 3}
        )
    )

    assert top_when_a_favored[PATH_A] > top_when_a_favored[PATH_B]
    assert top_when_b_favored[PATH_B] > top_when_b_favored[PATH_A]


def test_final_selection_tracks_edge_weight_only_as_exact_tie_breaker():
    """退化情形：历史 RMSE 位级精确相等时，边权确实能通过候选顺序左右 tie-break。

    根因不是"边权真的影响了打分"——两个候选的 error/latency/resource 完全
    相等（见下方断言），SLA 优化器的 np.argmin 在多个 Pareto 分数打平时取
    列表首个；而候选列表顺序由图谱推理按边权排序后 append，因此边权能通过
    "谁先入列"这个side channel 间接决定平局结果。这是一个脆弱的退化行为，
    不代表边权被真正计入了决策评分。
    """
    result_a_favored = _select(weight_a=0.95, weight_b=0.05, rmse_a=0.02, rmse_b=0.02)
    result_b_favored = _select(weight_a=0.05, weight_b=0.95, rmse_a=0.02, rmse_b=0.02)

    assert result_a_favored["metrics"]["error"] == result_b_favored["metrics"]["error"]
    assert result_a_favored["path_id"] == PATH_A
    assert result_b_favored["path_id"] == PATH_B


def test_final_selection_ignores_edge_weight_once_history_error_differs_at_all():
    """核心诊断：只要历史误差存在任何真实差异（哪怕 0.5%），边权无论多极端都不再改变选择。

    这是能解释 27 轮真实实验现象的关键结果：真实历史 RMSE 几乎不可能位级
    相等，所以生产链路里 tie-break side channel 几乎永远不会被触发——边权
    变化在实践中对最终选择没有可观测影响，不是因为信号丢失，而是它从未有
    机会在"error 已经能分出胜负"时改变胜负。
    """
    rmse_a, rmse_b = 0.0200, 0.0201  # 仅 0.5% 差异，model_a 略优

    result_a_favored = _select(weight_a=1.0, weight_b=0.0, rmse_a=rmse_a, rmse_b=rmse_b)
    result_b_favored = _select(weight_a=0.0, weight_b=1.0, rmse_a=rmse_a, rmse_b=rmse_b)

    assert result_a_favored["path_id"] == PATH_A
    assert result_b_favored["path_id"] == PATH_A
    assert result_a_favored["path_id"] == result_b_favored["path_id"]
