from src.core.enums import ModelLifecycleStage, TaskType
from src.core.index import IndexManager
from src.core.schema import ModelManifest
from src.core.solver import build_solver
from src.core.solver.backends import CombinatorBackend, ProtocolBBackend
from src.graph.model_graph import ModelGraph


def _manifest(model_id="m_ok"):
    return ModelManifest(
        model_id=model_id,
        task_types=[TaskType.FORECASTING],
        business_domains=["load_forecast"],
        input_constraints={"features": ["load"]},
        output_schema={"yhat": "float"},
        resource_cost={},
        lifecycle_stage=ModelLifecycleStage.ACTIVE,
    )


def test_build_solver_protocol_b_uses_default_backend_and_stages():
    manifests = {"m_ok": _manifest()}
    manager = IndexManager.with_defaults(
        manifests=manifests,
        scenario_records=[
            {
                "scenario_id": "s1",
                "signature": {"load": 1.0},
                "business_domain": "load_forecast",
            }
        ],
    )

    solver = build_solver("protocol_b", manifests=manifests, index_manager=manager)

    assert isinstance(solver.backend, ProtocolBBackend)
    assert len(solver.pre_stages) == 2
    assert solver.post_stages == []


def test_build_solver_combinator_uses_default_backend():
    solver = build_solver("combinator", manifests={"m_ok": _manifest()})

    assert isinstance(solver.backend, CombinatorBackend)
    assert len(solver.pre_stages) == 1


def test_build_solver_rejects_unknown_mode():
    try:
        build_solver("unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
        assert "protocol_b" in str(exc)
        assert "combinator" in str(exc)
    else:
        raise AssertionError("build_solver must reject unknown mode")


def test_build_solver_preserves_custom_stage_order():
    calls = []

    def pre(ctx, trace):
        calls.append("pre")

    def post(ctx, trace):
        calls.append("post")

    solver = build_solver("combinator", pre_stages=[pre], post_stages=[post])

    assert solver.pre_stages == [pre]
    assert solver.post_stages == [post]


def test_build_solver_can_attach_uncertainty_gate_before_custom_post_stages():
    class FakeGate:
        def apply(self, ctx, trace):
            pass

    def post(ctx, trace):
        pass

    gate = FakeGate()
    solver = build_solver(
        "combinator",
        uncertainty_gate=gate,
        post_stages=[post],
    )

    assert solver.post_stages[0].__self__ is gate
    assert solver.post_stages[0].__name__ == "apply"
    assert solver.post_stages[1] is post


def test_build_solver_attaches_cascade_after_uncertainty_gate():
    class FakeGate:
        def apply(self, ctx, trace):
            pass

    class FakeCascade:
        def apply(self, ctx, trace):
            pass

    gate = FakeGate()
    cascade = FakeCascade()

    solver = build_solver(
        "combinator",
        uncertainty_gate=gate,
        cascade_decider=cascade,
    )

    assert solver.post_stages[0].__self__ is gate
    assert solver.post_stages[1].__self__ is cascade


def test_build_solver_attaches_temporal_relation_stage_after_other_post_stages():
    class FakeGate:
        def apply(self, ctx, trace):
            pass

    class FakeCascade:
        def apply(self, ctx, trace):
            pass

    graph = ModelGraph()
    solver = build_solver(
        "combinator",
        uncertainty_gate=FakeGate(),
        cascade_decider=FakeCascade(),
        temporal_relation_graph=graph,
    )

    assert solver.post_stages[2].__name__ == "temporal_relation_stage"
