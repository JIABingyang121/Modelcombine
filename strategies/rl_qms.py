from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class StrategyOutput:
    y_pred_test: np.ndarray
    weights_log: Optional[np.ndarray]
    meta: Dict[str, Any]
    chosen_models: Optional[np.ndarray] = None

    def __iter__(self):
        """
        向后兼容旧调用：out, chosen = strategy.predict(...)
        统一接口后仍建议直接使用 StrategyOutput.chosen_models。
        """
        yield self
        yield self.chosen_models


class BaseStrategy:
    def fit(self, P_val, y_val, ctx_val=None, model_names=None):
        raise NotImplementedError

    def predict(self, P_test, y_test=None, ctx_test=None, model_names=None) -> StrategyOutput:
        raise NotImplementedError


def compute_em(P: np.ndarray, y: np.ndarray, metric: str = "ape", eps: float = 1e-6) -> np.ndarray:
    """
    P: [L, M], y: [L]
    return em: [L, M]
    DLF 的 EM 用 APE = |yhat - y| / y
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    if metric == "ape":
        denom = np.maximum(np.abs(y), eps)
        return np.abs(P - y[:, None]) / denom[:, None]
    if metric == "mae":
        return np.abs(P - y[:, None])
    if metric == "mse":
        return (P - y[:, None]) ** 2
    raise ValueError(f"unknown metric={metric}")


def compute_rank(em: np.ndarray) -> np.ndarray:
    """
    em: [L, M] 越小越好
    rk: [L, M] 1=最好
    """
    em = np.asarray(em, dtype=float)
    L, M = em.shape
    rk = np.zeros((L, M), dtype=np.int32)
    order = np.argsort(em, axis=1)
    for t in range(L):
        rk[t, order[t]] = np.arange(1, M + 1, dtype=np.int32)
    return rk


def train_q_table(
    rk: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.8,
    Ne: int = 100,
    seed: int = 0,
    switch_penalty: float = 0.0,  # 新增：切换惩罚
) -> np.ndarray:
    """
    rk: [Nq, M] 训练窗口内每时刻每模型的排名
    输出 Q: [M, M]
    reward 用排名改善：R = RK(Di,t) - RK(Dj,t+1) - switch_penalty * (s != a)
    """
    rk = np.asarray(rk, dtype=np.int32)
    Nq, M = rk.shape
    Q = np.zeros((M, M), dtype=np.float32)
    if Nq < 2 or M == 0 or Ne <= 0:
        return Q

    rng = np.random.default_rng(seed)
    eps = 1.0
    eps_decay = 1.0 / max(1, Ne)

    for _ in range(Ne):
        s = int(rng.integers(M))
        for t in range(Nq - 1):
            if rng.random() < eps:
                a = int(rng.integers(M))
            else:
                a = int(np.argmax(Q[s]))
            # 排名改善 + 切换惩罚
            r = float(rk[t, s] - rk[t + 1, a])
            if s != a:
                r -= switch_penalty
            s_next = a
            td = r + gamma * float(np.max(Q[s_next])) - float(Q[s, a])
            Q[s, a] = float(Q[s, a] + alpha * td)
            s = s_next
        eps = max(0.0, eps - eps_decay)

    return Q


def select_models(Q: np.ndarray, s0: int, n_steps: int, epsilon: float = 0.0, seed: int = 0) -> np.ndarray:
    """
    用 Q* 选 n_steps 步模型
    
    Args:
        Q: Q-table [M, M]
        s0: 初始状态
        n_steps: 步数
        epsilon: 探索率（测试时建议设为 0）
        seed: 随机种子
    
    Returns:
        chosen: 选择的模型索引数组
    """
    Q = np.asarray(Q, dtype=float)
    M = Q.shape[0]
    chosen = np.zeros((n_steps,), dtype=np.int32)
    if n_steps <= 0 or M == 0:
        return chosen
    
    rng = np.random.default_rng(seed)
    s = int(s0)
    for k in range(n_steps):
        # epsilon-greedy: 以 epsilon 概率随机探索
        if epsilon > 0 and rng.random() < epsilon:
            a = int(rng.integers(M))
        else:
            a = int(np.argmax(Q[s]))
        chosen[k] = a
        s = a
    return chosen


class RLQMSStrategy(BaseStrategy):
    """
    DLF-QMS 的工程化复现（改进版）：
    - 每 Nsp 步训练一个 Q-agent（最近 Nq 步 rank）
    - 用该 agent 为接下来 Nsp 步选择模型（one-hot）
    
    改进点：
    1. active_models: 候选池约束，只在活跃模型中选择
    2. switch_penalty: 切换惩罚，抑制频繁切换
    3. test_epsilon: 测试时的探索率（建议设为 0）
    4. Nq_multiplier: 根据 seasonal_period 自动调整 Nq
    """

    name = "rl_qms"

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.8,
        Ne: int = 100,
        Nq: int = 72,
        Nsp: int = 4,
        em_metric: str = "ape",
        seed: int = 0,
        warm_start_from_val: bool = True,
        # 新增参数
        active_models: list = None,  # 候选池约束，只在这些模型中选择
        switch_penalty: float = 0.0,  # 切换惩罚（reward -= switch_penalty if switch）
        test_epsilon: float = 0.0,    # 测试时的探索率（0=纯贪心）
        Nq_multiplier: float = 3.0,   # Nq = seasonal_period * Nq_multiplier
        seasonal_period: int = None,  # 若指定，Nq 会自动调整
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.Ne = Ne
        self.Nq = Nq
        self.Nsp = Nsp
        self.em_metric = em_metric
        self.seed = seed
        self.warm_start_from_val = warm_start_from_val
        
        # 新增参数
        self.active_models = active_models
        self.switch_penalty = switch_penalty
        self.test_epsilon = test_epsilon
        self.Nq_multiplier = Nq_multiplier
        self.seasonal_period = seasonal_period
        
        # 根据 seasonal_period 自动调整 Nq
        if seasonal_period is not None:
            self.Nq = int(seasonal_period * Nq_multiplier)

        self._P_hist0 = None
        self._y_hist0 = None
        self._fallback_id = None
        self._active_indices = None  # 活跃模型的索引映射
        self._full_to_active = None  # 全模型索引 -> 活跃模型索引
        self._active_to_full = None  # 活跃模型索引 -> 全模型索引

    def fit(self, P_val, y_val, ctx_val=None, model_names=None):
        """在验证集上拟合 RL-QMS 策略
        
        Args:
            P_val: 验证集预测矩阵 [N, M]
            y_val: 验证集真值
            ctx_val: 上下文特征（未使用）
            model_names: 模型名列表，用于匹配 active_models
        """
        if (P_val is not None) and (y_val is not None):
            P_val = np.asarray(P_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            if P_val.ndim == 1:
                P_val = P_val[:, None]
            
            M = P_val.shape[1]
            
            # 处理候选池约束
            if self.active_models is not None and model_names is not None:
                # 找出活跃模型的索引
                self._active_indices = [i for i, m in enumerate(model_names) if m in self.active_models]
                if not self._active_indices:
                    # 无匹配的活跃模型，使用全部模型
                    self._active_indices = list(range(M))
                self._active_to_full = {i: idx for i, idx in enumerate(self._active_indices)}
                self._full_to_active = {idx: i for i, idx in enumerate(self._active_indices)}
            else:
                # 无候选池约束，使用全部模型
                self._active_indices = list(range(M))
                self._active_to_full = {i: i for i in range(M)}
                self._full_to_active = {i: i for i in range(M)}
            
            # 只用活跃模型计算 warm start 和 fallback
            P_active = P_val[:, self._active_indices]
            
            if self.warm_start_from_val:
                tail = min(len(y_val), self.Nq)
                self._P_hist0 = P_active[-tail:].copy()
                self._y_hist0 = y_val[-tail:].copy()
            
            if len(P_active) == len(y_val) and P_active.size > 0:
                mae = np.mean(np.abs(P_active - y_val[:, None]), axis=0)
                self._fallback_id = int(np.argmin(mae))  # 在活跃模型中的索引
        return self

    def predict(
        self,
        P_test,
        y_test=None,
        ctx_test=None,
        model_names=None,
    ) -> StrategyOutput:
        if y_test is None:
            raise ValueError("RL-QMS 需要 y_test 做因果在线训练")

        P_test = np.asarray(P_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        if P_test.ndim == 1:
            P_test = P_test[:, None]
        if len(P_test) != len(y_test):
            raise ValueError(f"P_test/y_test length mismatch: {len(P_test)} vs {len(y_test)}")

        T, M = P_test.shape
        if T == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return StrategyOutput(
                y_pred_test=empty,
                weights_log=np.zeros((0, M), dtype=np.float32),
                meta={},
                chosen_models=empty.astype(np.int32),
            )

        # 处理候选池约束：只在活跃模型中选择
        if self._active_indices is None:
            # 未调用 fit 或无候选池约束
            self._active_indices = list(range(M))
            self._active_to_full = {i: i for i in range(M)}
            self._full_to_active = {i: i for i in range(M)}
        
        M_active = len(self._active_indices)
        
        # 只用活跃模型的预测
        P_test_active = P_test[:, self._active_indices]
        
        if self._P_hist0 is not None:
            # _P_hist0 已经是活跃模型子集
            P_hist = np.vstack([self._P_hist0, P_test_active])
            y_hist = np.concatenate([self._y_hist0, y_test])
            offset = len(self._y_hist0)
        else:
            P_hist, y_hist, offset = P_test_active, y_test, 0

        # 在活跃模型空间中选择
        chosen_models_active = np.zeros((T,), dtype=np.int32)
        chosen_models = np.zeros((T,), dtype=np.int32)  # 转换回全模型空间
        weights_log = np.zeros((T, M), dtype=np.float32)
        y_pred = np.zeros((T,), dtype=np.float32)

        block_meta = []
        for t0 in range(0, T, self.Nsp):
            t1 = min(T, t0 + self.Nsp)
            end = offset + t0
            start = max(0, end - self.Nq)
            fallback_used = False
            if end - start < 2:
                fallback_used = True
                fallback_id = 0 if self._fallback_id is None else int(self._fallback_id)
                chosen = np.full((t1 - t0,), fallback_id, dtype=np.int32)
            else:
                P_tr = P_hist[start:end]
                y_tr = y_hist[start:end]
                em = compute_em(P_tr, y_tr, metric=self.em_metric)
                rk = compute_rank(em)
                Q = train_q_table(
                    rk,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    Ne=self.Ne,
                    seed=self.seed + t0,
                    switch_penalty=self.switch_penalty,  # 传入切换惩罚
                )
                if t0 > 0:
                    s0 = int(chosen_models_active[t0 - 1])
                else:
                    s0 = int(np.argmin(em[-1]))
                # 使用 test_epsilon 作为测试时探索率
                chosen = select_models(Q, s0=s0, n_steps=(t1 - t0), 
                                       epsilon=self.test_epsilon, seed=self.seed + t0)

            for i, tt in enumerate(range(t0, t1)):
                m_id_active = int(chosen[i])  # 活跃模型空间的索引
                m_id_full = self._active_to_full[m_id_active]  # 转换到全模型空间
                chosen_models_active[tt] = m_id_active
                chosen_models[tt] = m_id_full
                weights_log[tt, m_id_full] = 1.0
                y_pred[tt] = float(P_test[tt, m_id_full])

            switches = int(np.sum(chosen[1:] != chosen[:-1])) if len(chosen) > 1 else 0
            block_meta.append(
                {
                    "t0": int(t0),
                    "t1": int(t1),
                    "start": int(start),
                    "end": int(end),
                    "fallback_used": bool(fallback_used),
                    "models_used_in_block": int(len(np.unique(chosen))),
                    "switches_in_block": switches,
                }
            )

        meta = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "Ne": self.Ne,
            "Nq": self.Nq,
            "Nsp": self.Nsp,
            "em_metric": self.em_metric,
            "switch_penalty": self.switch_penalty,
            "test_epsilon": self.test_epsilon,
            "active_models_count": M_active,
            "total_models_count": M,
            "models_used": int(len(np.unique(chosen_models))),
            "switch_rate": float(np.mean(chosen_models[1:] != chosen_models[:-1])) if T > 1 else 0.0,
            "cost_assumption": "posthoc_selection",
            "block_meta": block_meta,
        }

        return StrategyOutput(y_pred, weights_log, meta, chosen_models=chosen_models)
