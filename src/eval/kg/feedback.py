"""Closed-loop feedback store for KG ensemble.

After each (dataset, horizon) Protocol B run the **validation** MAE of the
selected model combination is written back via ``write_back()``.  The stored
quality scores are then used in the next horizon's ``apply_to_graph()`` call
to apply a **multiplicative boost** on existing complementary edge weights.

Design decisions
----------------
* **Val-only signal** (fixes test-contamination): feedback uses val_mae_b and
  val_mae_a (Protocol A reference).  Test labels are never seen.
* **Normalized quality score ∈ [0, 1], neutral at 0.5**: the signal is
  ``quality = clip(1 - 0.5 * (val_mae_b / val_mae_a), 0, 1)``.
  Tie (B == A) → 0.5 (neutral); B perfect → 1.0 (max boost); B 2× worse → 0.0.
  This aligns with apply_to_graph's multiplier formula (neutral point = 0.5).
  pair_w (harmonic mean of Ridge weights) modulates the EMA learning rate only.
* **Multiplicative boost** in apply_to_graph instead of EMA-toward-tiny-value:
  ``new_w = clip(current_w * (1 + lr * (2·score - 1)), 0, 1)``
  so score=1 → boost, score=0.5 → neutral, score=0 → slight reduction.
* **Bounded history** to prevent unbounded pkl growth.
* **Config hash** on save/load for cross-session consistency guard.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class KGFeedbackStore:
    """跨步长闭环反馈存储器。

    Parameters
    ----------
    learning_rate : float
        反馈注入强度，默认 0.1。  越大则历史信号影响越强。
    max_history : int
        history 日志条目上限，超出后 FIFO 淘汰，默认 200。
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_history: int = 200,
    ) -> None:
        self.lr: float = float(learning_rate)
        self.max_history: int = int(max_history)
        # {(m1, m2): quality_score ∈ [0,1]}
        self.complementary_scores: Dict[Tuple[str, str], float] = {}
        # {model: quality_score ∈ [0,1]}
        self.model_scores: Dict[str, float] = {}
        # bounded FIFO audit log
        self._history: deque = deque(maxlen=self.max_history)

    # ------------------------------------------------------------------
    # Public: Write path
    # ------------------------------------------------------------------

    def write_back(
        self,
        *,
        selected: List[str],
        weights: Dict[str, float],
        val_mae_b: float,
        val_mae_a: float,
        horizon: int,
        dataset: str,
    ) -> None:
        """将 val-set 误差写回图谱权重（不使用 test 标签）。

        Parameters
        ----------
        selected : list[str]
            Protocol B 选定的模型（guard 触发前的 B 候选，由调用方选择）。
        weights : dict[str, float]
            对应的 Ridge 权重（原始，无需归一化）。
        val_mae_b : float
            Protocol B 的 val MAE。
        val_mae_a : float
            Protocol A 的 val MAE（参照基线，用于归一化质量信号）。
        horizon : int
            预测步长。
        dataset : str
            数据集名称。
        """
        if not selected:
            return
        if not (np.isfinite(val_mae_b) and val_mae_b > 0):
            return
        if not (np.isfinite(val_mae_a) and val_mae_a > 0):
            return

        # 质量信号 ∈ [0, 1]，中性点与 apply_to_graph 的乘法公式对齐（score=0.5 为中性）：
        #   quality = clip(1.0 - 0.5 * (val_mae_b / val_mae_a), 0, 1)
        #   val_mae_b == val_mae_a       → quality = 0.5（持平 → 中性，不增强也不削弱）
        #   val_mae_b == 0.5 * val_mae_a → quality = 0.75（B 好 50% → 轻度增强）
        #   val_mae_b == 0               → quality = 1.0（B 完美 → 最强增强）
        #   val_mae_b == 1.5 * val_mae_a → quality = 0.25（B 差 50% → 轻度削弱）
        #   val_mae_b >= 2 * val_mae_a   → quality = 0.0（B 差 2× → 最强削弱）
        quality = float(np.clip(1.0 - 0.5 * (val_mae_b / val_mae_a), 0.0, 1.0))

        # 归一化权重：sum(|w|) = 1
        total_w = sum(abs(weights.get(m, 0.0)) for m in selected) or 1.0

        # 1. 单模型得分 EMA
        # model_score 直接使用 quality（∈ [0,1]），与 pair_scores 尺度一致。
        # w（归一化 Ridge 权重）仅调节有效 EMA 学习率：高贡献模型更新更快。
        for m in selected:
            w = abs(weights.get(m, 1.0 / len(selected))) / total_w
            model_lr = float(np.clip(self.lr * (1.0 + w), 0.0, 1.0))
            if m in self.model_scores:
                self.model_scores[m] = (
                    (1.0 - model_lr) * self.model_scores[m] + model_lr * quality
                )
            else:
                self.model_scores[m] = quality

        # 2. 互补对得分 EMA（pair_score = quality ∈ [0, 1]）
        # 注意：pair_score 直接使用 quality，不乘以 pair_w。
        # 若乘以 pair_w（调和平均，归一化权重下最大 ≈ 0.5），pair_score 永远 ≤ 0.5，
        # 而 apply_to_graph 的乘法公式以 score=0.5 为中性点，score > 0.5 才能触发增强——
        # 这会导致"boost"机制实际上只能削弱边权。
        # 修复：pair_w 仅作为 EMA 有效学习率的调节因子：高贡献对更新更快。
        for i, m1 in enumerate(selected):
            for m2 in selected[i + 1 :]:
                w1 = abs(weights.get(m1, 0.0)) / total_w
                w2 = abs(weights.get(m2, 0.0)) / total_w
                # pair_w ∈ (0, 0.5]：调和平均，两权重越均衡值越大
                pair_w = 2.0 * w1 * w2 / (w1 + w2 + 1e-9)
                pair_score = quality  # ∈ [0, 1]，与边权尺度匹配
                # pair_w 调节有效学习率：高贡献对学得稍快，上限为 1.5×base
                pair_lr = float(np.clip(self.lr * (1.0 + pair_w), 0.0, 1.0))
                key = _pair_key(m1, m2)
                if key in self.complementary_scores:
                    self.complementary_scores[key] = (
                        (1.0 - pair_lr) * self.complementary_scores[key]
                        + pair_lr * pair_score
                    )
                else:
                    self.complementary_scores[key] = pair_score

        # 3. 有界 FIFO 日志
        self._history.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "selected": list(selected),
                "val_mae_b": float(val_mae_b),
                "val_mae_a": float(val_mae_a),
                "quality": quality,
            }
        )

    # ------------------------------------------------------------------
    # Public: Read path
    # ------------------------------------------------------------------

    def apply_to_graph(self, mg: Any) -> Dict[str, Any]:
        """将历史质量分以乘法 boost 方式注入 ModelGraph 互补边权。

        boost 公式（edge weight 保持在 [0, 1]）：
            new_w = clip(current_w * (1 + lr * (2·score - 1)), 0, 1)
        解读：
            score = 1.0  → new_w = current_w * (1 + lr)   （增强）
            score = 0.5  → new_w ≈ current_w               （中性）
            score = 0.0  → new_w = current_w * (1 - lr)   （轻微压制）

        Parameters
        ----------
        mg : ModelGraph
            当前 Protocol B 的图实例（就地修改互补边权）。
        """
        updated = 0
        skipped = 0
        for (m1, m2), score in self.complementary_scores.items():
            if not np.isfinite(score):
                skipped += 1
                continue
            score = float(np.clip(score, 0.0, 1.0))
            multiplier = 1.0 + self.lr * (2.0 * score - 1.0)
            for src, tgt in [(m1, m2), (m2, m1)]:
                if mg.G.has_edge(src, tgt):
                    current = float(mg.G[src][tgt].get("weight", 0.5))
                    mg.G[src][tgt]["weight"] = float(
                        np.clip(current * multiplier, 0.0, 1.0)
                    )
                    updated += 1
                else:
                    skipped += 1
        return {"updated_edges": updated, "skipped_edges": skipped}

    # ------------------------------------------------------------------
    # Public: Query helpers
    # ------------------------------------------------------------------

    def get_pair_score(self, m1: str, m2: str) -> Optional[float]:
        return self.complementary_scores.get(_pair_key(m1, m2))

    def get_model_score(self, model: str) -> Optional[float]:
        return self.model_scores.get(model)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_model_scores": len(self.model_scores),
            "n_pair_scores": len(self.complementary_scores),
            "n_history_entries": len(self._history),
            "top_model_scores": sorted(
                self.model_scores.items(), key=lambda kv: kv[1], reverse=True
            )[:5],
            "top_pair_scores": sorted(
                self.complementary_scores.items(), key=lambda kv: kv[1], reverse=True
            )[:5],
        }

    # ------------------------------------------------------------------
    # Persistence with config consistency guard
    # ------------------------------------------------------------------

    def save(self, path: str, config_fingerprint: Optional[str] = None) -> None:
        """序列化到磁盘，可附带配置指纹用于跨 session 一致性校验。

        Parameters
        ----------
        config_fingerprint : str, optional
            调用方传入的配置哈希（如模型池 hash）。保存到文件，load 时比对。
        """
        data = {
            "complementary_scores": self.complementary_scores,
            "model_scores": self.model_scores,
            "history": list(self._history),
            "learning_rate": self.lr,
            "max_history": self.max_history,
            "config_fingerprint": config_fingerprint,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(
            f"[feedback] saved to {path} "
            f"({len(self.model_scores)} models, "
            f"{len(self.complementary_scores)} pairs, "
            f"fp={config_fingerprint})"
        )

    def load(
        self,
        path: str,
        config_fingerprint: Optional[str] = None,
    ) -> bool:
        """从磁盘加载，若配置指纹不匹配则拒绝加载并返回 False。

        Parameters
        ----------
        config_fingerprint : str, optional
            当前运行的配置哈希。与文件中保存的哈希不一致时拒绝加载。
            传入 None 则跳过一致性校验。

        Returns
        -------
        bool
            True 表示加载成功，False 表示跳过（文件不存在或指纹不符）。
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            # 配置一致性校验
            # 若调用方提供了 fingerprint，则：
            #   saved_fp 不一致（包括旧文件无 fingerprint 的 None）→ 拒绝加载。
            # 只有 config_fingerprint 为 None 时才跳过校验（宽松模式）。
            saved_fp = data.get("config_fingerprint")
            if config_fingerprint is not None:
                if saved_fp != config_fingerprint:
                    print(
                        f"[feedback] config fingerprint mismatch "
                        f"(saved={saved_fp!r}, current={config_fingerprint!r}), "
                        f"skipping load to avoid stale experience injection"
                    )
                    return False

            self.complementary_scores = data.get("complementary_scores", {})
            self.model_scores = data.get("model_scores", {})
            loaded_history = data.get("history", [])
            self._history = deque(loaded_history[-self.max_history:], maxlen=self.max_history)
            # 注意：不从文件恢复 learning_rate，始终保留调用方实例化时的值。
            # 若覆盖，外部通过 env/参数设定的 lr 会被历史文件静默改写。
            print(
                f"[feedback] loaded from {path} "
                f"({len(self.model_scores)} models, "
                f"{len(self.complementary_scores)} pairs)"
            )
            return True
        except Exception as e:
            print(f"[feedback] load failed ({e}), starting fresh")
            return False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_config_fingerprint(model_pool: List[str], extra: Optional[Dict] = None) -> str:
    """为模型池生成配置指纹，用于跨 session 一致性校验。"""
    payload: Dict[str, Any] = {"model_pool": sorted(model_pool)}
    if extra:
        payload.update(extra)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _pair_key(m1: str, m2: str) -> Tuple[str, str]:
    return (m1, m2) if m1 <= m2 else (m2, m1)
