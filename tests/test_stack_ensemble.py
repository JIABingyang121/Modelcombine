"""Multi-layer stack ensemble 复现版（src/models/stack_ensemble.py）。

依据 arXiv:2511.15350v1 §3.2 / §4.1 / §4.2 / §B.2。这里逐条锁住论文里定义明确的算法，
用可以手算的小例子，而不是"跑通即可"。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.stack_ensemble import (
    GREEDY_ITERATIONS,
    K_FOLDS,
    MultiLayerStackEnsemble,
    Stacker,
    _greedy_ensemble_weights,
    _performance_weights,
    build_layer2,
)


def test_greedy_ensemble_follows_caruana_with_replacement():
    """§B.2 式 3：带放回、等权，第 j 步权重为 ((j-1)ω + e_m)/j。

    两个模型：A 完美、B 有偏。贪心必须每一步都选 A，S 步之后权重为 (1, 0)。
    """
    y = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.column_stack([y, y + 10.0])

    for iterations in (1, 10, 100):
        weights = _greedy_ensemble_weights(predictions, y, iterations)
        np.testing.assert_allclose(weights, [1.0, 0.0], atol=0)

    # 两个模型对称地偏离真值时，带放回的等权集成应收敛到各 0.5
    symmetric = np.column_stack([y - 1.0, y + 1.0])
    weights = _greedy_ensemble_weights(symmetric, y, 10)
    np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-12)
    assert weights.sum() == pytest.approx(1.0)


def test_performance_weighted_averages_match_the_three_h_functions():
    """§B.2：先把损失归一化到和为 1，再过 h，再把权重归一化。"""
    losses = np.array([1.0, 3.0])
    normalised = losses / losses.sum()  # [0.25, 0.75]

    for mode, h in (
        ("inv", lambda L: 1.0 / L),
        ("sqr", lambda L: 1.0 / L ** 2),
        ("exp", lambda L: np.exp(1.0 / L)),
    ):
        expected = h(normalised) / h(normalised).sum()
        np.testing.assert_allclose(_performance_weights(losses, mode), expected, rtol=1e-12)
        assert _performance_weights(losses, mode).sum() == pytest.approx(1.0)

    # 损失更小的模型必须拿到更大的权重
    assert _performance_weights(losses, "inv")[0] > _performance_weights(losses, "inv")[1]
    with pytest.raises(ValueError, match="未知的性能加权模式"):
        _performance_weights(losses, "cube")


def test_layer2_covers_the_implemented_families():
    names = [s.name for s in build_layer2()]
    assert names == [
        "mean", "median", "model_selection",
        "performance_weighted_inv", "performance_weighted_sqr", "performance_weighted_exp",
        *[f"greedy_{n}" for n in GREEDY_ITERATIONS],
    ]


def test_model_selection_picks_the_best_validation_model_and_uses_only_it():
    y = np.array([1.0, 2.0, 3.0])
    train = np.column_stack([y + 5.0, y])          # 第二个模型完美
    stacker = Stacker("model_selection", "model_selection").fit(train, y)

    assert stacker.selected_ == 1
    probe = np.column_stack([np.zeros(3), np.array([7.0, 8.0, 9.0])])
    np.testing.assert_allclose(stacker.predict(probe), [7.0, 8.0, 9.0])


def test_mean_and_median_ignore_validation_data():
    y = np.array([0.0, 0.0])
    train = np.column_stack([y, y, y])
    probe = np.column_stack([np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([9.0, 9.0])])

    np.testing.assert_allclose(
        Stacker("mean", "mean").fit(train, y).predict(probe), [4.0, 4.0]
    )
    np.testing.assert_allclose(
        Stacker("median", "median").fit(train, y).predict(probe), [2.0, 2.0]
    )


def _folds(n_folds: int, n_points: int = 8):
    rng = np.random.default_rng(0)
    folds = []
    for _ in range(n_folds):
        y = rng.normal(100.0, 5.0, n_points)
        good = y + rng.normal(0, 0.2, n_points)
        bad = y + rng.normal(0, 8.0, n_points)
        folds.append((np.column_stack([good, bad]), y))
    return folds


def test_two_level_cross_validation_fits_l3_on_the_last_fold_and_refits_l2_on_all():
    """§4.2：L2 先用前 K-1 折训练、在第 K 折上产出 L3 的训练输入，最后 L2 用全部 K 折重训。

    可观测的后果是：L2 的最终权重必须等于"用全部 K 折拟合"的结果，而不是前 K-1 折的结果。
    """
    # 前 K-1 折 A 更准、第 K 折 B 明显更准：只有把第 K 折也算进来，L2 的权重才会改变。
    # 这样"用全部 K 折重训"与"只用前 K-1 折"才是可区分的，断言不至于落空。
    folds = []
    for index in range(K_FOLDS):
        y = np.linspace(100.0, 110.0, 8)
        if index < K_FOLDS - 1:
            folds.append((np.column_stack([y + 1.0, y + 3.0]), y))
        else:
            folds.append((np.column_stack([y + 10.0, y + 0.1]), y))
    ensemble = MultiLayerStackEnsemble().fit(folds)

    all_predictions = np.vstack([p for p, _y in folds])
    all_targets = np.concatenate([y for _p, y in folds])
    head_predictions = np.vstack([p for p, _y in folds[:-1]])
    head_targets = np.concatenate([y for _p, y in folds[:-1]])

    greedy = next(s for s in ensemble.layer2 if s.name == "greedy_100")
    expected_all = _greedy_ensemble_weights(all_predictions, all_targets, 100)
    expected_head = _greedy_ensemble_weights(head_predictions, head_targets, 100)
    np.testing.assert_allclose(greedy.weights_, expected_all, atol=1e-12)
    assert not np.allclose(expected_all, expected_head), "装置需让两者可区分，否则断言为空"

    # L3 是在 L2 输出上拟合的贪心集成，权重非负且和为 1
    assert ensemble.layer3.weights_ is not None
    assert len(ensemble.layer3.weights_) == len(ensemble.layer2)
    assert ensemble.layer3.weights_.min() >= 0
    assert ensemble.layer3.weights_.sum() == pytest.approx(1.0)


def test_prediction_is_a_composition_of_l2_then_l3():
    folds = _folds(K_FOLDS)
    ensemble = MultiLayerStackEnsemble().fit(folds)
    base = np.column_stack([np.arange(6, dtype=float), np.arange(6, dtype=float) + 3.0])

    layer2_output = np.column_stack([s.predict(base) for s in ensemble.layer2])
    np.testing.assert_allclose(
        ensemble.predict(base), ensemble.layer3.predict(layer2_output), atol=0
    )
    # 三层组合必须落在基预测的凸包内（全部 L2/L3 都是凸组合或选择）
    assert ensemble.predict(base).min() >= base.min() - 1e-9
    assert ensemble.predict(base).max() <= base.max() + 1e-9


def test_two_level_cv_needs_at_least_two_folds():
    with pytest.raises(ValueError, match="至少需要 2 折"):
        MultiLayerStackEnsemble().fit(_folds(1))


# ------------------------------------------- exp 加权的数值稳定性（溢出回归）
#
# `exp` 模式原本是 `np.exp(1.0 / normalised)`。某个候选的损失远小于其余时，它的
# normalised 份额趋近 0，`1/normalised` 会到 1e8 量级，`exp` 直接溢出成 inf，
# 归一化后整条权重退化成 [nan, 0, ...]——正式实验里这会让该折的组合器彻底失效。
# 现在改为数学等价的 softmax 平移形式：分子分母同乘 exp(-max) 完全抵消。

def test_exp_weights_do_not_overflow_on_extreme_loss_ratios():
    """一个候选比其余好 8 个数量级时仍必须给出有效权重。"""
    losses = np.array([1e-8, 1.0, 1.0])

    weights = _performance_weights(losses, "exp")

    assert np.isfinite(weights).all(), f"权重含非有限值: {weights}"
    assert not np.isnan(weights).any()
    # 反向确认装置有效：旧公式在这组输入上确实溢出
    with np.errstate(over="ignore"):
        legacy = np.exp(1.0 / (losses / losses.sum()))
    assert np.isinf(legacy).any(), "装置无效：这组输入没有触发旧公式的溢出"


@pytest.mark.parametrize("losses", [
    np.array([1e-12, 1.0, 1.0]),
    np.array([1e-8, 1e-8, 1.0]),
    np.array([1.0, 1e6, 1e12]),
    np.array([1e-6, 1e-3, 1.0, 1e3]),
])
def test_exp_weights_stay_finite_non_negative_and_sum_to_one(losses):
    weights = _performance_weights(losses, "exp")

    assert np.isfinite(weights).all()
    assert (weights >= 0).all(), f"权重不得为负: {weights}"
    assert weights.sum() == pytest.approx(1.0)


def test_exp_weights_are_monotone_in_loss():
    """损失越小权重越高——这是加权平均的定义，不能被数值处理改掉。"""
    losses = np.array([0.1, 0.2, 0.4, 0.8])

    weights = _performance_weights(losses, "exp")

    assert np.all(np.diff(weights) < 0), f"权重应随损失单调下降: {weights}"
    assert weights.argmax() == int(losses.argmin())


def test_exp_weights_match_the_original_formula_on_ordinary_values():
    """不溢出的普通量级上，新旧公式必须逐值相同——这次改的是数值实现，不是口径。"""
    for losses in (
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([1.0, 1.5]),
        np.array([2.0, 4.0, 4.0, 8.0]),
    ):
        legacy_raw = np.exp(1.0 / (losses / np.sum(losses)))
        assert np.isfinite(legacy_raw).all(), "该用例本就不该溢出，否则测不到等价性"
        expected = legacy_raw / legacy_raw.sum()

        np.testing.assert_allclose(
            _performance_weights(losses, "exp"), expected, rtol=1e-12, atol=0
        )
