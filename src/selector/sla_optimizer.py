"""
SLA 多目标优化器
实现 Pareto 前沿求解和拉格朗日多目标优化
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize, LinearConstraint
from dataclasses import dataclass


@dataclass
class SLAConstraints:
    """SLA 约束条件"""
    max_latency: float = float('inf')  # 最大延迟 (ms)
    max_memory: float = float('inf')   # 最大内存 (MB)
    max_cpu: float = 100.0             # 最大 CPU (%)
    min_accuracy: float = 0.0          # 最小准确度要求


@dataclass
class PathCandidate:
    """路径候选"""
    path_id: str
    models: List[str]
    error_estimate: float      # 预估误差
    latency_estimate: float    # 预估延迟
    resource_estimate: float   # 预估资源消耗
    metadata: Dict[str, Any] = None


class ParetoOptimizer:
    """Pareto 前沿优化器"""
    
    def __init__(self):
        self.pareto_front: List[PathCandidate] = []
    
    def compute_pareto_front(self, candidates: List[PathCandidate]) -> List[PathCandidate]:
        """计算 Pareto 前沿
        
        Args:
            candidates: 候选路径列表
            
        Returns:
            Pareto 最优解集合
        """
        if not candidates:
            return []
        
        # 提取目标向量 (error, latency, resource) - 越小越好
        objectives = np.array([
            [c.error_estimate, c.latency_estimate, c.resource_estimate]
            for c in candidates
        ])
        
        # 计算支配关系
        pareto_mask = np.ones(len(candidates), dtype=bool)
        
        for i in range(len(candidates)):
            if not pareto_mask[i]:
                continue
            
            # 检查 i 是否被任何其他点支配
            for j in range(len(candidates)):
                if i == j or not pareto_mask[j]:
                    continue
                
                # j 支配 i 当且仅当：j 在所有目标上不差于 i，且至少一个目标严格更好
                dominates = np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])
                
                if dominates:
                    pareto_mask[i] = False
                    break
        
        pareto_front = [candidates[i] for i in range(len(candidates)) if pareto_mask[i]]
        self.pareto_front = pareto_front
        
        print(f"  [Pareto优化] 从 {len(candidates)} 个候选中找到 {len(pareto_front)} 个非支配解")
        
        return pareto_front
    
    def select_from_pareto(self, pareto_front: List[PathCandidate], 
                          weights: Dict[str, float]) -> PathCandidate:
        """从 Pareto 前沿中根据偏好权重选择最优解
        
        Args:
            pareto_front: Pareto 前沿候选集
            weights: 目标权重 {'error': w1, 'latency': w2, 'resource': w3}
            
        Returns:
            最优路径候选
        """
        if not pareto_front:
            raise ValueError("Pareto 前沿为空")
        
        if len(pareto_front) == 1:
            return pareto_front[0]
        
        # 归一化目标值
        objectives = np.array([
            [c.error_estimate, c.latency_estimate, c.resource_estimate]
            for c in pareto_front
        ])
        
        obj_min = objectives.min(axis=0) + 1e-9
        obj_max = objectives.max(axis=0) + 1e-9
        objectives_norm = (objectives - obj_min) / (obj_max - obj_min + 1e-9)
        
        # 加权求和
        w = np.array([weights['error'], weights['latency'], weights['resource']])
        scores = np.dot(objectives_norm, w)
        
        best_idx = np.argmin(scores)  # 最小化加权目标
        return pareto_front[best_idx]


class LagrangianOptimizer:
    """拉格朗日多目标优化器（带约束）"""
    
    def __init__(self, constraints: SLAConstraints):
        self.constraints = constraints
    
    def optimize(self, candidates: List[PathCandidate], 
                weights: Dict[str, float]) -> Tuple[PathCandidate, Dict[str, float]]:
        """使用拉格朗日乘数法求解带约束的最优路径
        
        Args:
            candidates: 候选路径列表
            weights: 初始权重偏好
            
        Returns:
            (最优路径, 最优权重)
        """
        if not candidates:
            raise ValueError("候选路径为空")
        
        # 过滤满足硬约束的候选
        feasible = [
            c for c in candidates
            if c.latency_estimate <= self.constraints.max_latency
            and c.resource_estimate <= self.constraints.max_memory
        ]
        
        if not feasible:
            print(f"  [拉格朗日优化] 无可行解，放宽约束")
            feasible = candidates
        
        print(f"  [拉格朗日优化] 可行解: {len(feasible)}/{len(candidates)}")
        
        # 构建优化问题：min f(x) = w1*e + w2*l + w3*r
        # s.t. l <= max_latency, r <= max_memory
        
        # 提取目标矩阵
        objectives = np.array([
            [c.error_estimate, c.latency_estimate, c.resource_estimate]
            for c in feasible
        ])
        
        # 归一化
        obj_min = objectives.min(axis=0) + 1e-9
        obj_max = objectives.max(axis=0) + 1e-9
        objectives_norm = (objectives - obj_min) / (obj_max - obj_min + 1e-9)
        
        # 定义目标函数（最小化加权和）
        def objective_func(w):
            """目标函数：加权多目标"""
            scores = np.dot(objectives_norm, w)
            return scores.min()  # 最坏情况最小化
        
        # 约束：权重和为1，权重非负
        constraints_opt = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 和为1
        ]
        bounds = [(0.0, 1.0)] * 3  # 权重范围 [0, 1]
        
        # 初始权重
        w0 = np.array([weights['error'], weights['latency'], weights['resource']])
        w0 = w0 / np.sum(w0)  # 归一化
        
        # 求解优化问题
        try:
            result = minimize(
                objective_func, w0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_opt,
                options={'ftol': 1e-6, 'maxiter': 100}
            )
            
            if result.success:
                optimal_weights = result.x
                print(f"  [拉格朗日优化] 最优权重: Error={optimal_weights[0]:.3f}, "
                      f"Latency={optimal_weights[1]:.3f}, Resource={optimal_weights[2]:.3f}")
            else:
                optimal_weights = w0
                print(f"  [拉格朗日优化] 优化失败，使用初始权重")
        except Exception as e:
            print(f"  [拉格朗日优化] 优化异常: {e}，使用初始权重")
            optimal_weights = w0
        
        # 使用最优权重选择最佳候选
        scores = np.dot(objectives_norm, optimal_weights)
        best_idx = np.argmin(scores)
        best_candidate = feasible[best_idx]
        
        optimal_weights_dict = {
            'error': float(optimal_weights[0]),
            'latency': float(optimal_weights[1]),
            'resource': float(optimal_weights[2])
        }
        
        # 打印优化详情，增强解释性
        print(f"  [SLA优化详情] 候选数: {len(candidates)}, 可行解: {len(feasible)}")
        print(f"  [SLA优化详情] 最优权重: {optimal_weights_dict}")
        print(f"  [SLA优化详情] 选中路径: {best_candidate.path_id}")
        print(f"  [SLA优化详情] 预估指标: Error={best_candidate.error_estimate:.4f}, "
              f"Latency={best_candidate.latency_estimate:.1f}ms, Resource={best_candidate.resource_estimate:.1f}MB")
        
        return best_candidate, optimal_weights_dict


class SLAOptimizerManager:
    """SLA 优化管理器（集成 Pareto 和拉格朗日方法）"""
    
    def __init__(self, constraints: SLAConstraints = None, mode: str = "pareto"):
        """
        Args:
            constraints: SLA 约束条件
            mode: 优化模式 ['pareto', 'lagrangian', 'hybrid']
        """
        self.constraints = constraints or SLAConstraints()
        self.mode = mode
        self.pareto_optimizer = ParetoOptimizer()
        self.lagrangian_optimizer = LagrangianOptimizer(self.constraints)
    
    def optimize(self, candidates: List[PathCandidate], 
                weights: Dict[str, float]) -> Tuple[PathCandidate, Dict[str, float], Dict[str, Any]]:
        """执行 SLA 优化
        
        Args:
            candidates: 候选路径列表
            weights: 权重偏好
            
        Returns:
            (最优路径, 使用的权重, 优化详情)
        """
        if not candidates:
            raise ValueError("候选路径为空")
        
        # 优化详情（用于调试和反馈学习）
        optimize_info = {
            'mode': self.mode,
            'total_candidates': len(candidates),
            'pareto_size': 0,
            'feasible_count': len(candidates),
            'constraint_slack': {}  # 约束余量
        }
        
        if self.mode == "pareto":
            # Pareto 前沿 + 偏好选择
            pareto_front = self.pareto_optimizer.compute_pareto_front(candidates)
            optimize_info['pareto_size'] = len(pareto_front)
            best_candidate = self.pareto_optimizer.select_from_pareto(pareto_front, weights)
            self._log_optimize_details(optimize_info, best_candidate, weights)
            return best_candidate, weights, optimize_info
        
        elif self.mode == "lagrangian":
            # 拉格朗日优化（自动调整权重）
            best_candidate, final_weights = self.lagrangian_optimizer.optimize(candidates, weights)
            optimize_info['pareto_size'] = len(candidates)
            self._log_optimize_details(optimize_info, best_candidate, final_weights)
            return best_candidate, final_weights, optimize_info
        
        elif self.mode == "hybrid":
            # 混合模式：先 Pareto 筛选，再拉格朗日优化
            pareto_front = self.pareto_optimizer.compute_pareto_front(candidates)
            optimize_info['pareto_size'] = len(pareto_front)
            
            if len(pareto_front) <= 3:
                # Pareto 前沿较小，直接选择
                best_candidate = self.pareto_optimizer.select_from_pareto(pareto_front, weights)
                self._log_optimize_details(optimize_info, best_candidate, weights)
                return best_candidate, weights, optimize_info
            else:
                # Pareto 前沿较大，使用拉格朗日优化
                best_candidate, final_weights = self.lagrangian_optimizer.optimize(pareto_front, weights)
                self._log_optimize_details(optimize_info, best_candidate, final_weights)
                return best_candidate, final_weights, optimize_info
        
        else:
            raise ValueError(f"未知优化模式: {self.mode}")
    
    def _log_optimize_details(self, info: Dict[str, Any], best: PathCandidate, weights: Dict[str, float]) -> None:
        """输出优化详情日志"""
        # 计算约束余量
        if self.constraints:
            info['constraint_slack'] = {
                'latency_slack': self.constraints.max_latency - best.latency_estimate if self.constraints.max_latency < float('inf') else float('inf'),
                'memory_slack': self.constraints.max_memory - best.resource_estimate if self.constraints.max_memory < float('inf') else float('inf')
            }
        
        print(f"  [SLAOptimizer] 模式={info['mode']}, 候选数={info['total_candidates']}, Pareto集={info['pareto_size']}")
        print(f"  [SLAOptimizer] 最终权重: Error={weights.get('error', 0):.3f}, Latency={weights.get('latency', 0):.3f}, Resource={weights.get('resource', 0):.3f}")
        if info['constraint_slack']:
            print(f"  [SLAOptimizer] 约束余量: Latency余量={info['constraint_slack'].get('latency_slack', 'N/A'):.1f}ms, Memory余量={info['constraint_slack'].get('memory_slack', 'N/A'):.1f}MB")
