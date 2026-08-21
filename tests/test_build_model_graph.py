import src.pipeline.main as pipeline_main
from src.pipeline.main import PowerPredictionPipeline


def _pipeline_with_assets(assets_cfg):
    pipeline = PowerPredictionPipeline.__new__(PowerPredictionPipeline)
    pipeline.config = {"assets": assets_cfg}
    return pipeline


def _minimal_assets_cfg():
    return {
        "models": [
            {"id": "prophet", "input_constraints": {"features": []}},
            {"id": "lgbm_reg", "input_constraints": {"features": []}},
        ],
        "relations": [],
        "selection_rules": [
            {
                "scenario_filter": {"region_type": "residential"},
                "prefer": ["prophet", "lgbm_reg"],
            }
        ],
    }


def test_build_model_graph_does_not_raise(tmp_path, monkeypatch):
    # 隔离 reports/graph_state.pkl 持久化图谱，保证测试密封
    monkeypatch.setattr(pipeline_main, "PROJECT_ROOT", str(tmp_path))
    pipeline = _pipeline_with_assets(_minimal_assets_cfg())

    mg = pipeline.build_model_graph()

    assert mg is not None


def test_build_model_graph_connects_scenario_to_real_path_hyperedges(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_main, "PROJECT_ROOT", str(tmp_path))
    pipeline = _pipeline_with_assets(_minimal_assets_cfg())

    mg = pipeline.build_model_graph()

    scenario_nodes = [
        n for n, d in mg.G.nodes(data=True) if d.get("node_type") == "scenario"
    ]
    assert len(scenario_nodes) == 1
    scenario_id = scenario_nodes[0]

    recommended = [
        (target, data)
        for _, target, data in mg.G.out_edges(scenario_id, data=True)
        if data.get("edge_type") == "recommended_for"
    ]
    assert len(recommended) == 2

    recommended_models = set()
    for path_id, _ in recommended:
        path_node = mg.G.nodes[path_id]
        assert path_node.get("node_type") == "path"
        assert path_node.get("is_hyperedge") is True
        assert len(path_node.get("members", [])) == 1
        recommended_models.add(path_node["members"][0])

    assert recommended_models == {"prophet", "lgbm_reg"}
