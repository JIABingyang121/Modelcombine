"""Protocol B 最终组合推理的可序列化重放器。

Protocol B 的最终预测可能是：
- 纯线性组合 ``X[members] @ weights``（Ridge，或被 post-adjustment 覆盖后的
  ``w_adj``）；
- 线性组合 + interaction 残差（interaction 分支被接受、且其后的 post-adjustment
  未被接受时）。

只保存模型编号和线性权重不足以重放第二种情况，因此这里保存 interaction 残差
回归器的已拟合状态与 feature 居中均值。设计矩阵列顺序严格按引擎构造顺序：
外层遍历 feature、内层遍历 member。

预测器由 ``src/eval/kg/protocol_b.py`` 在拟合这些对象的同一执行路径上构造，
不从 trace 事后猜测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

COMBINATION_PREDICTOR_KEY = "_combination_predictor"


@dataclass
class InteractionResidual:
    """interaction 残差分支的已拟合状态。

    ``columns`` 与设计矩阵的列一一对应，每项是 ``(member_id, feature_name)``。
    ``feature_means`` 是每个 feature 在 validation 上的居中均值（引擎用它对
    val 和 test 同时居中）。``regressor`` 是已拟合的 Ridge（``fit_intercept=False``）。
    """

    columns: List[Tuple[str, str]]
    feature_means: Mapping[str, float]
    regressor: Any

    def _design_matrix(
        self,
        base_predictions: Mapping[str, Sequence[float]],
        raw_features: pd.DataFrame,
    ) -> np.ndarray:
        cols = []
        for member_id, feature_name in self.columns:
            mean = float(self.feature_means[feature_name])
            feat = pd.to_numeric(raw_features[feature_name], errors="coerce")
            feat = feat.fillna(mean).to_numpy(dtype=float) - mean
            cols.append(np.asarray(base_predictions[member_id], dtype=float) * feat)
        return np.column_stack(cols)

    def apply(
        self,
        base_predictions: Mapping[str, Sequence[float]],
        raw_features: pd.DataFrame,
    ) -> np.ndarray:
        design = self._design_matrix(base_predictions, raw_features)
        return np.asarray(self.regressor.predict(design), dtype=float)


@dataclass
class CombinationPredictor:
    """离线拟合、在线重放的 Protocol B 组合最终预测器。"""

    member_ids: List[str]
    linear_weights: List[float]
    strategy: str
    interaction: Optional[InteractionResidual] = None

    def predict(
        self,
        base_predictions: Mapping[str, Sequence[float]],
        raw_features: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """由各成员在预测数据上的候选预测（和 interaction 所需原始特征）产生最终预测。"""
        stacked = np.column_stack(
            [np.asarray(base_predictions[m], dtype=float) for m in self.member_ids]
        )
        yhat = stacked @ np.asarray(self.linear_weights, dtype=float)
        if self.interaction is not None:
            if raw_features is None:
                raise ValueError(
                    "combination predictor has an interaction term but raw_features is None"
                )
            yhat = yhat + self.interaction.apply(base_predictions, raw_features)
        return yhat

    @property
    def required_model_ids(self) -> List[str]:
        return list(self.member_ids)

    @property
    def required_feature_names(self) -> List[str]:
        return sorted(self.interaction.feature_means) if self.interaction is not None else []
