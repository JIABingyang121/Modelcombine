"""方案 §3.5「必须先写的真实入口测试」：用户预测长度成为一等契约。

六条契约逐条对应文档表格：预测长度契约、同场景长度隔离、完整轨迹选择、
成员独立递归、离线—在线真实接线、无兼容长度。全部走真实建库脚本与真实
``run.py predict`` 子进程，不打桩。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models.artifacts import load_artifact
from src.models.trajectory_forecast import (
    TrajectoryForecastError,
    calendar_frame,
    future_timestamps,
    generate_member_trajectory,
)
from src.storage.model_store import ModelStore
from tests.forecast_steps_fixtures import (
    BASE_HORIZON,
    run_build,
    run_predict,
    task_of,
    write_history,
    write_scenario,
)

ALL_STEPS = [24, 168, 720]
ROWS = 2200


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return run_build(tmp_path_factory.mktemp("library"), ALL_STEPS, rows=ROWS)


def _predict(library, tmp_path, forecast_steps, *, name=""):
    task = task_of(library["report"], forecast_steps)
    history = write_history(
        tmp_path, library["frames"]["test"], task["test_origin"], f"history{name}.csv"
    )
    scenario = write_scenario(tmp_path, forecast_steps=forecast_steps, name=f"scenario{name}.json")
    output = tmp_path / f"forecast{name}.csv"
    proc = run_predict(library["db"], scenario, history, output)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    trace = json.loads(output.with_suffix(".trace.json").read_text())
    return task, pd.read_csv(output), trace


# ---------------------------------------------------------------- 预测长度契约
@pytest.mark.parametrize("forecast_steps", ALL_STEPS)
def test_requested_forecast_steps_decides_output_length(library, tmp_path, forecast_steps):
    _task, forecast, trace = _predict(library, tmp_path, forecast_steps)

    assert len(forecast) == forecast_steps
    assert trace["forecast_steps"] == forecast_steps
    assert trace["n_rows"] == forecast_steps
    assert forecast["yhat"].notna().all()
    assert trace["selector_invoked"] is False
    assert trace["base_horizon"] == BASE_HORIZON


# ------------------------------------------------------------ 同场景长度隔离
def test_relations_are_isolated_by_forecast_steps(library, tmp_path):
    traces = {}
    for forecast_steps in ALL_STEPS:
        task, _forecast, trace = _predict(
            library, tmp_path, forecast_steps, name=f"_iso{forecast_steps}"
        )
        assert trace["relation_id"] == task["relation_id"]
        assert trace["scenario_id"] == task["scenario_id"]
        traces[forecast_steps] = trace

    # 三种长度必须落在三条互不相同的关系上，绝不允许 168 命中 24/720 的关系
    assert len({t["relation_id"] for t in traces.values()}) == 3
    assert len({t["scenario_id"] for t in traces.values()}) == 3

    store = ModelStore(str(library["db"]))
    for forecast_steps, trace in traces.items():
        assert store.get_scenario(trace["scenario_id"])["forecast_steps"] == forecast_steps
    store.close()


# -------------------------------------------------------------- 完整轨迹选择
def _best_by(task, metric):
    scores = task["candidate_validation_mae"]
    return min(scores, key=lambda name: scores[name][metric])


@pytest.mark.parametrize("forecast_steps", ALL_STEPS)
def test_selected_combination_minimises_full_trajectory_validation_mae(library, forecast_steps):
    """选择目标必须是整条 validation 轨迹的 MAE，而不是 test，也不是单点。"""
    task = task_of(library["report"], forecast_steps)
    enumerated = task["enumerated_combinations"]

    assert enumerated, "组合枚举结果不能为空"
    best = min(enumerated, key=lambda entry: (entry["validation_mae"], entry["members"]))
    assert task["validation_mae"] == pytest.approx(best["validation_mae"], abs=1e-12)
    assert [m.split("__")[-1] for m in task["effective_members"]] == best["members"]

    # test 只在冻结后评价：枚举结果里根本不带 test 指标，选择无从看到它
    assert all("test_mae" not in entry for entry in enumerated)


def test_full_trajectory_selection_drops_recursively_divergent_member(library):
    """lgbm_reg 是 gain=1.002 的持久化模型：单步最优，递归 720 步按 1.002^k 放大。

    单点 validation 目标在每个长度上都会选中它；完整轨迹目标必须在 720 步上
    把它排除。这就是方案 §2.2 所说的"h=1 单点表现最好 != 连续递归最稳定"。
    """
    short = task_of(library["report"], 24)
    long = task_of(library["report"], 720)

    # 单点目标在两个长度上都选它
    assert _best_by(short, "lead1_mae") == "lgbm_reg"
    assert _best_by(long, "lead1_mae") == "lgbm_reg"

    # 完整轨迹目标在 720 步上换人，并把它从安全候选里筛掉
    assert _best_by(long, "trajectory_mae") != "lgbm_reg"
    assert "lgbm_reg" in long["filter_excluded"]

    short_types = {m.split("__")[-1] for m in short["effective_members"]}
    long_types = {m.split("__")[-1] for m in long["effective_members"]}
    assert "lgbm_reg" in short_types
    assert "lgbm_reg" not in long_types


def test_forecast_steps_changes_the_selected_members(library):
    """同一地区、同一时期，预测长度不同则最优组合成员不同（方案 RQ2 的前提）。"""
    selected = {
        steps: tuple(task_of(library["report"], steps)["effective_members"])
        for steps in ALL_STEPS
    }
    assert len(set(selected.values())) > 1, selected


def test_candidate_requiring_future_exogenous_input_is_not_eligible(library):
    """xgboost_reg 需要未来 temp：在离线建库边界判定无候选资格，并写明原因。"""
    for forecast_steps in ALL_STEPS:
        task = task_of(library["report"], forecast_steps)
        assert "xgboost_reg" not in {m.split("__")[-1] for m in task["effective_members"]}
        skipped = {entry["model_type"]: entry["reason"] for entry in task["skipped_candidates"]}
        assert "xgboost_reg" in skipped
        assert "temp" in skipped["xgboost_reg"]


def test_trajectory_generation_rejects_underivable_features():
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=200, freq="h"),
            "load": np.linspace(100, 120, 200),
        }
    )
    with pytest.raises(TrajectoryForecastError, match="temp"):
        generate_member_trajectory(
            model=object(), model_type="xgboost_reg",
            required_features=["hour", "temp"],
            history=history, forecast_steps=24, country="US",
        )


# -------------------------------------------------------------- 成员独立递归
def test_members_recurse_independently_of_the_combination_output(library, tmp_path):
    """任一成员的预测历史都不得改变其他成员的轨迹，组合输出不回灌。

    在 24 步任务上做——它的最优组合有两个成员，单成员组合测不出成员间隔离。
    """
    _task, forecast, trace = _predict(library, tmp_path, 24, name="_indep")
    assert len(trace["member_types"]) >= 2, "本用例需要多成员组合才有意义"

    history = pd.read_csv(tmp_path / "history_indep.csv")
    history["timestamp"] = pd.to_datetime(history["timestamp"])

    store = ModelStore(str(library["db"]))
    rows = [store.get_model(model_id) for model_id in trace["model_ids"]]
    store.close()

    standalone = {
        member_type: generate_member_trajectory(
            model=load_artifact(row["artifact_path"]),
            model_type=row["model_type"],
            required_features=row["required_features"],
            history=history, forecast_steps=24, country="US",
        )
        for row, member_type in zip(rows, trace["member_types"])
    }
    # 各成员轨迹互不相同：断言不是在比较两条同样的线
    assert not np.allclose(*list(standalone.values())[:2])

    predictor = load_artifact(trace["artifact_paths"]["combination"])
    calendar = calendar_frame(pd.to_datetime(forecast["timestamp"]), "US")
    np.testing.assert_allclose(
        forecast["yhat"].to_numpy(dtype=float),
        predictor.predict(standalone, calendar),
        rtol=0, atol=1e-8,
    )

    # V2 的共同回灌行为（§2.3）：每步把组合输出写回同一份历史，所有成员下一步
    # 都读这个组合输出。当前输出必须与它显著不同，否则回灌并没有真正去掉。
    shared = history[["timestamp", "load"]].copy()
    models = [load_artifact(row["artifact_path"]) for row in rows]
    shared_output = []
    for step in range(24):
        base = {}
        for row, model, member_type in zip(rows, models, trace["member_types"]):
            base[member_type] = generate_member_trajectory(
                model=model, model_type=row["model_type"],
                required_features=row["required_features"],
                history=shared, forecast_steps=1, country="US",
            )
        step_ts = pd.to_datetime(forecast["timestamp"]).iloc[step]
        yhat = float(predictor.predict(base, calendar_frame([step_ts], "US"))[0])
        shared_output.append(yhat)
        shared.loc[len(shared)] = [step_ts, yhat]

    assert not np.allclose(
        forecast["yhat"].to_numpy(dtype=float),
        np.asarray(shared_output, dtype=float),
        rtol=0, atol=1e-6,
    )


# ------------------------------------------------------ 离线—在线真实接线
@pytest.mark.parametrize("forecast_steps", ALL_STEPS)
def test_offline_build_and_online_predict_agree_within_1e_8(library, tmp_path, forecast_steps):
    task, forecast, _trace = _predict(library, tmp_path, forecast_steps, name=f"_wire{forecast_steps}")

    np.testing.assert_allclose(
        forecast["yhat"].to_numpy(dtype=float),
        np.asarray(task["test_trajectory"], dtype=float),
        rtol=0, atol=1e-8,
    )
    assert forecast["timestamp"].tolist() == list(task["test_target_timestamps"])


# ------------------------------------------------------------------ 无兼容长度
@pytest.mark.parametrize("forecast_steps", [48, 720.5, "week", 0])
def test_unsupported_forecast_steps_exits_nonzero(library, tmp_path, forecast_steps):
    task = task_of(library["report"], 24)
    history = write_history(tmp_path, library["frames"]["test"], task["test_origin"], "h_bad.csv")
    scenario = write_scenario(
        tmp_path, forecast_steps=forecast_steps, name=f"bad_{forecast_steps}.json"
    )
    output = tmp_path / f"bad_{forecast_steps}.csv"

    proc = run_predict(library["db"], scenario, history, output)

    assert proc.returncode != 0
    assert "forecast_steps" in (proc.stdout + proc.stderr)
    assert not output.exists()


def test_missing_forecast_steps_relation_exits_nonzero(library, tmp_path):
    """长度受支持但库中没有该长度关系时，非零退出，不回退到别的长度。"""
    task = task_of(library["report"], 24)
    history = write_history(tmp_path, library["frames"]["test"], task["test_origin"], "h_gap.csv")
    scenario = write_scenario(tmp_path, forecast_steps=168, name="gap.json", region="aemo_vic")
    output = tmp_path / "gap.csv"

    proc = run_predict(library["db"], scenario, history, output)

    assert proc.returncode != 0
    assert "no compatible scenario" in (proc.stdout + proc.stderr).lower()
    assert not output.exists()


# ------------------------------------------------- 成员的真实多步语义（ARIMA/Prophet）
def _hourly_training_series(n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    t = np.arange(n)
    load = 1000.0 + 60.0 * np.sin(2 * np.pi * t / 24) + 0.6 * t
    return pd.DataFrame({"timestamp": ts, "load": load}, index=ts), pd.Series(load, index=ts)


def _user_history(n: int = 200) -> pd.DataFrame:
    ts = pd.date_range("2026-03-01", periods=n, freq="h")
    t = np.arange(n)
    return pd.DataFrame(
        {"timestamp": ts, "load": 900.0 + 40.0 * np.sin(2 * np.pi * t / 24) + 0.3 * t}
    )


def test_arima_is_ineligible_because_it_cannot_consume_user_history():
    """ARIMA 的 predict 只按 len(X) 从训练序列末尾外推。

    逐行调用会让每一步都取"训练末尾之后第 1 步"，退化成常数轨迹；它也没有消费
    用户最新历史的接口。第一版直接判无资格，不建在线更新框架。
    """
    from src.models.registry import model_registry

    frame, y = _hourly_training_series()
    model = model_registry.create("arima", order=(1, 1, 1), freq="h")
    model.fit(frame[["load"]], y)

    # 真实多步与"重复四次单步"不同——这正是逐行递归会丢掉的信息
    four_steps = np.asarray(model.predict(pd.DataFrame(index=range(4))), dtype=float)
    repeated_one_step = np.asarray(
        [float(model.predict(pd.DataFrame(index=range(1)))[0]) for _ in range(4)], dtype=float
    )
    assert len(set(np.round(four_steps, 6))) > 1
    assert np.allclose(repeated_one_step, repeated_one_step[0])

    with pytest.raises(TrajectoryForecastError, match="arima"):
        generate_member_trajectory(
            model=model, model_type="arima", required_features=["hour"],
            history=_user_history(), forecast_steps=24, country="US",
        )


def test_prophet_receives_all_target_timestamps_in_one_call():
    """Prophet 一次输出目标时间戳上的整条轨迹，不是逐点调用的常数。"""
    from src.models.registry import model_registry

    frame, y = _hourly_training_series()
    model = model_registry.create("prophet", daily_seasonality=True, weekly_seasonality=False)
    model.fit(frame[["load"]], y)

    history = _user_history()
    trajectory = generate_member_trajectory(
        model=model, model_type="prophet", required_features=["hour"],
        history=history, forecast_steps=24, country="US",
    )

    assert len(trajectory) == 24
    assert np.isfinite(trajectory).all()
    # 逐行调用的退化结果：每一步都回到 train_end + 1，是一条常数轨迹
    degenerate = np.asarray(
        [float(model.predict(pd.DataFrame(index=range(1)))[0]) for _ in range(24)], dtype=float
    )
    assert np.allclose(degenerate, degenerate[0])
    assert not np.allclose(trajectory, trajectory[0])
    assert not np.allclose(trajectory, degenerate)

    # 轨迹必须落在用户历史之后的目标时间戳上，而不是训练序列末尾之后
    expected_ts = future_timestamps(history, 24)
    replay = np.asarray(
        model.predict(pd.DataFrame(index=pd.DatetimeIndex(expected_ts))), dtype=float
    )
    np.testing.assert_allclose(trajectory, replay, rtol=0, atol=1e-12)
