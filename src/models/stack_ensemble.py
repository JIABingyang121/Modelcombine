"""Multi-layer stack ensemble 的**复现版**（reproduction），不是官方实现。

依据：Bosch, Shchur, Erickson, Bohlke-Schneider, Türkmen,
*Multi-layer Stack Ensembles for Time Series Forecasting*, AutoML 2025
（arXiv:2511.15350v1，本机副本 ``docs/0905/2511.15350v1.pdf``）。

三层结构（§4.1，Fig. 2）::

    y_{1:T} -> s( g_1(ŷ_1..ŷ_M), ..., g_C(ŷ_1..ŷ_M) )

- L1 = 基预测器 ``{f_m}``，输出 M 条 H 步轨迹；
- L2 = 一组 stacker ``{g_c}``，各自把 M 条基预测合成一条；
- L3 = 聚合器 ``s``，把 C 条 L2 预测合成最终输出。

两级时序交叉验证（§3.2 + §4.2）::

    1. L1 在 K 折时序 CV 上产生折外预测：第 k 折去掉最后 j=(K-k+1) 个长度 H 的窗口，
       在 y_{1:T-jH} 上训练，再做 H 步预测覆盖验证窗口 y^k = y_{T-jH:T-(j-1)H}
    2. L2 只用**前 K-1 个**验证窗口训练
    3. L2 在**第 K 个**窗口上预测，用这批预测 + 真值拟合 L3
    4. L2 再用**全部 K 个**窗口重训，以便用上最新数据

**与论文的两处偏差**（本项目实验方案所要求，必须随结果一起报告）：

1. 论文每折都从头重训 L1；本方案 §5 要求基预测器在 T1 之前只训练一次并全程冻结，
   因此这里用**冻结的 L1** 在 K 个验证窗口上生成预测，不逐折重训。
2. 论文 L2 用了 31 个组合方法、多层时限定 14 个；本复现实现其中定义明确的 9 个
   （2 简单平均 + 1 模型选择 + 3 性能加权 + 3 贪心集成），不含论文的 18 个线性变体
   与 4 个非线性表格模型。已实现的部分严格按 §B.2 的定义。

点预测口径（§6）：Q=1，只有中位数一条；权重按验证窗口上的 MAE 拟合。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

#: 论文 §6 Setup：L1 用 K=5 折时序交叉验证。
K_FOLDS = 5
#: 论文 §B.2：贪心集成的迭代次数三种取值。
GREEDY_ITERATIONS = (10, 100, 1000)


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(yhat - y)))


def _greedy_ensemble_weights(
    predictions: np.ndarray, y: np.ndarray, iterations: int
) -> np.ndarray:
    """Caruana et al. (2004) 的带放回贪心集成（§B.2 式 3）。

    从零权重出发，第 j 步选使 ``((j-1)ω^(j-1) + e_m)/j`` 损失最小的模型 m。
    """
    n_models = predictions.shape[1]
    weights = np.zeros(n_models, dtype=float)
    for step in range(1, iterations + 1):
        best_index, best_loss = 0, np.inf
        for m in range(n_models):
            candidate = ((step - 1) * weights + np.eye(n_models)[m]) / step
            loss = _mae(y, predictions @ candidate)
            if loss < best_loss:
                best_index, best_loss = m, loss
        weights = ((step - 1) * weights + np.eye(n_models)[best_index]) / step
    return weights


def _performance_weights(losses: np.ndarray, mode: str) -> np.ndarray:
    """§B.2 性能加权平均：先把验证损失归一化到和为 1，再过 h，再归一化权重。"""
    normalised = losses / np.sum(losses)
    if mode == "inv":
        raw = 1.0 / normalised
    elif mode == "sqr":
        raw = 1.0 / np.square(normalised)
    elif mode == "exp":
        # 与 exp(1/normalised) 数学等价的稳定形式：softmax 对分子分母同乘 exp(-max)
        # 后完全抵消，但避免了 exp 溢出。某个候选损失远小于其余时，1/normalised 会
        # 达到 1e8 量级，直接 exp 会得到 inf，归一化后整条权重退化成 [nan, 0, ...]。
        scores = 1.0 / normalised
        raw = np.exp(scores - np.max(scores))
    else:
        raise ValueError(f"未知的性能加权模式: {mode}")
    return raw / np.sum(raw)


class Stacker:
    """一个 L2/L3 组合器：``fit`` 在折外数据上定权，``predict`` 合成一条预测。"""

    def __init__(self, name: str, kind: str, **options: object) -> None:
        self.name = name
        self.kind = kind
        self.options = options
        self.weights_: np.ndarray | None = None
        self.selected_: int | None = None

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "Stacker":
        n_models = predictions.shape[1]
        if self.kind in ("mean", "median"):
            self.weights_ = None
        elif self.kind == "model_selection":
            losses = [_mae(y, predictions[:, m]) for m in range(n_models)]
            self.selected_ = int(np.argmin(losses))
        elif self.kind == "performance_weighted":
            losses = np.array(
                [_mae(y, predictions[:, m]) for m in range(n_models)], dtype=float
            )
            self.weights_ = _performance_weights(losses, str(self.options["mode"]))
        elif self.kind == "greedy":
            self.weights_ = _greedy_ensemble_weights(
                predictions, y, int(self.options["iterations"])
            )
        else:
            raise ValueError(f"未知的 stacker 类型: {self.kind}")
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        if self.kind == "mean":
            return predictions.mean(axis=1)
        if self.kind == "median":
            return np.median(predictions, axis=1)
        if self.kind == "model_selection":
            return predictions[:, self.selected_]
        return predictions @ self.weights_


def build_layer2() -> List[Stacker]:
    """§B.2 中定义明确、且在本复现实现范围内的 9 个组合方法。"""
    stackers = [
        Stacker("mean", "mean"),
        Stacker("median", "median"),
        Stacker("model_selection", "model_selection"),
    ]
    stackers += [
        Stacker(f"performance_weighted_{mode}", "performance_weighted", mode=mode)
        for mode in ("inv", "sqr", "exp")
    ]
    stackers += [
        Stacker(f"greedy_{n}", "greedy", iterations=n) for n in GREEDY_ITERATIONS
    ]
    return stackers


class MultiLayerStackEnsemble:
    """论文 §4 的多层堆叠，L3 用贪心集成聚合 L2（论文的 multi-layer stacking 变体）。"""

    def __init__(self, layer3_iterations: int = 100) -> None:
        self.layer2: List[Stacker] = build_layer2()
        self.layer3 = Stacker("greedy_l3", "greedy", iterations=layer3_iterations)

    def fit(self, folds: Sequence[Tuple[np.ndarray, np.ndarray]]) -> "MultiLayerStackEnsemble":
        """``folds`` 是 K 个 ``(基预测矩阵 [n_points, M], 真值 [n_points])``。

        按 §4.2：L2 先只用前 K-1 折训练 -> 在第 K 折上预测并据此拟合 L3 ->
        L2 再用全部 K 折重训。
        """
        if len(folds) < 2:
            raise ValueError(f"两级交叉验证至少需要 2 折，收到 {len(folds)}")
        head_predictions = np.vstack([p for p, _y in folds[:-1]])
        head_targets = np.concatenate([y for _p, y in folds[:-1]])
        for stacker in self.layer2:
            stacker.fit(head_predictions, head_targets)

        last_predictions, last_targets = folds[-1]
        layer2_on_last = np.column_stack(
            [stacker.predict(last_predictions) for stacker in self.layer2]
        )
        self.layer3.fit(layer2_on_last, last_targets)

        all_predictions = np.vstack([p for p, _y in folds])
        all_targets = np.concatenate([y for _p, y in folds])
        for stacker in self.layer2:
            stacker.fit(all_predictions, all_targets)
        return self

    def predict(self, base_predictions: np.ndarray) -> np.ndarray:
        layer2_output = np.column_stack(
            [stacker.predict(base_predictions) for stacker in self.layer2]
        )
        return self.layer3.predict(layer2_output)
