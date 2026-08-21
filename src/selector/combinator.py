from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from ..models.implementations import WeightedBlender, StackingBlender
from .scenario_similarity import PowerScenarioAnalyzer
from .adaptive_weights import AdaptiveWeightManager
from .resource_predictor import ResourcePredictor
from .sla_optimizer import SLAOptimizerManager, SLAConstraints, PathCandidate


class PowerModelCombinator:
    """电力模型组合器，专门用于电力需求预测的模型选择和组合"""
    
    # 特征组定义：将抽象特征名映射到实际列名
    FEATURE_GROUPS = {
        "weather": {"temp", "temperature", "humidity", "wind", "rain", "feels_like_temp", "weather_comfort"},
        "time_features": {"hour", "dow", "dayofweek", "month", "year", "day", "is_weekend"},
        "holidays": {"is_holiday", "holiday", "is_workday", "pre_holiday", "post_holiday"},
        "categorical_features": {"region_type", "region_id", "region"},
        "region_id": {"region", "region_id"}
    }
    
    def __init__(self, enable_adaptive_weights: bool = True, enable_resource_prediction: bool = True):
        self.scenario_analyzer = PowerScenarioAnalyzer()
        self.historical_performance = {}  # 存储历史模型性能
        self.model_compatibility = {}     # 存储模型兼容性信息
        self._historical_scenarios_data = []  # 历史场景完整数据（用于路径评分）
        
        # Phase 2 新增：轻量级学习模块
        self.enable_adaptive_weights = enable_adaptive_weights
        self.enable_resource_prediction = enable_resource_prediction
        
        # Phase 3 新增：SLA 优化器
        self.sla_optimizer = SLAOptimizerManager(mode="hybrid")
        
        if enable_adaptive_weights:
            self.weight_manager = AdaptiveWeightManager(mode="hybrid")
        else:
            self.weight_manager = None
        
        if enable_resource_prediction:
            self.resource_predictor = ResourcePredictor()
        else:
            self.resource_predictor = None
        
    def set_historical_scenarios(self, hist_scenarios: List[Tuple[str, Dict[str, float], Dict[str, float]]]):
        """设置历史场景数据（供路径评分使用）"""
        self._historical_scenarios_data = hist_scenarios
        
        # 训练资源预测器 (Phase 2: 基于历史数据回归)
        if self.resource_predictor and hist_scenarios:
            training_data = []
            for _, feats, perf in hist_scenarios:
                # 尝试从性能数据中提取资源指标
                metrics = {}
                # 优先使用 _profiling 数据
                if '_profiling' in perf and isinstance(perf['_profiling'], dict):
                    prof = perf['_profiling']
                    # 兼容不同键名 (latency/latency_ms, memory/memory_mb)
                    metrics['latency_ms'] = prof.get('latency', prof.get('latency_ms', prof.get('elapsed', 0)))
                    metrics['memory_mb'] = prof.get('memory', prof.get('memory_mb', 0))
                    metrics['cpu_percent'] = prof.get('cpu', prof.get('cpu_percent', 0))
                else:
                    # 回退到顶层指标
                    # 注：resource 字段含义不明确，可能是综合消耗或内存，此处作为 memory 的回退
                    if 'latency' in perf: metrics['latency_ms'] = perf['latency']
                    if 'resource' in perf: metrics['memory_mb'] = perf['resource']
                
                # 为缺失字段设置合理默认值（避免0值污染训练集）
                # 默认值基于典型模型的经验估计
                if metrics.get('latency_ms', 0) <= 0:
                    metrics['latency_ms'] = 100.0  # 默认 100ms
                if metrics.get('memory_mb', 0) <= 0:
                    metrics['memory_mb'] = 50.0   # 默认 50MB
                if metrics.get('cpu_percent', 0) <= 0:
                    metrics['cpu_percent'] = 10.0  # 默认 10%
                    
                # 提取模型ID (假设 perf 中有 model_id 或者它是单模型场景)
                model_id = perf.get('model_id', 'unknown')
                
                # 估算数据规模 (从特征中推断或使用默认值)
                data_size = int(feats.get('data_size', 24 * 30)) # 默认一个月小时数
                num_features = len(feats)
                
                training_data.append((model_id, data_size, num_features, metrics))
            
            if training_data:
                self.resource_predictor.fit(training_data)
    
    def _filter_by_feature_completeness(self, available_models: List[str], 
                                       actual_data_columns: set,
                                       model_graph: Any = None) -> List[str]:
        """过滤掉特征不完备的模型（严格校验：逐项检查模型所需特征是否在数据列中）
        
        Args:
            available_models: 候选模型列表
            actual_data_columns: 实际数据框中的列名集合
            model_graph: 模型知识图谱
            
        Returns:
            特征完备的模型列表
        """
        if not model_graph:
            return available_models
        
        filtered = []
        for model_id in available_models:
            # 跳过非实际模型（如Blender）
            if 'blender' in model_id.lower():
                continue
            
            # 获取模型的特征需求
            input_constraints = model_graph.get_node_attr(model_id, "input_constraints", {})
            required_features = input_constraints.get("features", [])
            
            # 如果没有明确的特征需求，默认通过
            if not required_features:
                filtered.append(model_id)
                continue
            
            # 逐项校验：所有required_features必须在actual_data_columns中
            required_set = set(required_features)
            
            # 扩展校验逻辑：支持特征组
            missing = set()
            for req in required_set:
                if req in actual_data_columns:
                    continue
                
                # 检查是否为特征组
                if req in self.FEATURE_GROUPS:
                    # 如果特征组中的任意一个特征存在，则认为满足条件
                    group_features = self.FEATURE_GROUPS[req]
                    if not group_features.intersection(actual_data_columns):
                        missing.add(req)
                else:
                    missing.add(req)
            
            if not missing:
                filtered.append(model_id)
            else:
                print(f"      跳过模型 {model_id}: 缺失特征 {missing}")
        
        return filtered if filtered else available_models  # 兜底：如果全部过滤，返回原列表
        
    def select_optimal_path(self, scenario_signature: Dict[str, float], 
                          available_models: List[str],
                          constraints: Dict[str, float] = None,
                          model_graph: Any = None,
                          similar_scenarios: List[Tuple[str, float]] = None,
                          actual_data_columns: set = None) -> Dict[str, Any]:
        """
        多目标约束下的最优路径选择（增强版：基于历史场景检索 + 严格特征校验）
        
        Args:
            scenario_signature: 场景特征
            available_models: 可用模型列表
            constraints: 约束条件 (max_latency, max_cost, etc.)
            model_graph: 模型知识图谱 (ModelGraph)
            similar_scenarios: 相似历史场景 [(scenario_id, similarity), ...]
            actual_data_columns: 实际数据框列名集合（用于严格校验特征完备性）
            
        Returns:
            Dict: 最优路径配置 {"path_id": str, "models": List[str], "weights": Dict, "strategy": str}
        """
        constraints = constraints or {}
        
        # 1. 特征完备性过滤：移除缺少必需特征的模型（严格模式）
        if actual_data_columns is None:
            # 兜底：如果未传入实际列，使用scenario_signature的keys
            actual_data_columns = set(scenario_signature.keys()) if scenario_signature else set()
        
        filtered_models = self._filter_by_feature_completeness(available_models, actual_data_columns, model_graph)
        if not filtered_models:
            print("      警告: 所有模型均被特征校验过滤，使用兜底策略")
            filtered_models = available_models  # 兜底
        
        # 2. 生成候选路径 (基于图谱推理 + 启发式规则)
        # 从 scenario_signature 中提取 scenario_id (如果有)
        scenario_id = scenario_signature.get('_scenario_id') if scenario_signature else None
        candidates = self._generate_candidate_paths(
            filtered_models, model_graph, 
            scenario_id=scenario_id, 
            available_features=actual_data_columns
        )
        
        # 3. 评估每条路径的性能和代价
        evaluated_paths = []
        for path in candidates:
            # 使用历史场景数据进行真实性能评估
            cost_metrics = self._evaluate_path_cost(path, scenario_signature, model_graph, similar_scenarios)
            
            # 检查约束
            if constraints.get("max_latency") and cost_metrics["latency"] > constraints["max_latency"]:
                continue
            if constraints.get("max_resource") and cost_metrics["resource"] > constraints["max_resource"]:
                continue
            
            evaluated_paths.append({
                "path": path,
                "metrics": cost_metrics
            })
            
        # 4. 选择最优路径
        if not evaluated_paths:
            # 兜底策略：返回简单的平均组合
            return {
                "path_id": "fallback_avg",
                "models": filtered_models[:3],
                "weights": {m: 1.0/3 for m in filtered_models[:3]},
                "strategy": "weighted_mean"
            }

        # 额外安全网：记录最优单模型（作为组合退化时的回退）
        single_candidates = [p for p in evaluated_paths if len(p["path"].get("models", [])) == 1]
        best_single = None
        if single_candidates:
            best_single = min(single_candidates, key=lambda x: x["metrics"]["error"])
            
        # ========== SLA/多目标优化: Pareto + 拉格朗日求解 (Phase 3) ==========
        # 【Phase 3 升级】
        #   - 引入 Pareto 前沿筛选非支配解
        #   - 使用拉格朗日乘数法处理硬约束
        #   - 结合自适应权重进行最终决策
        # ============================================================
        
        # 1. 获取基础偏好权重
        if self.enable_adaptive_weights and self.weight_manager:
            region_type_code = scenario_signature.get("region_type", 0.0)
            scenario_type = {1.0: "residential", 2.0: "charging", 3.0: "service_area"}.get(region_type_code, "default")
            adaptive_weights = self.weight_manager.get_adaptive_weights(scenario_signature, scenario_type)
        else:
            adaptive_weights = {"error": 0.6, "latency": 0.2, "resource": 0.2}
            
        # 2. 构建优化候选集
        path_candidates = []
        for p in evaluated_paths:
            m = p["metrics"]
            candidate = PathCandidate(
                path_id=p["path"]["path_id"],
                models=p["path"]["models"],
                error_estimate=m["error"],
                latency_estimate=m["latency"],
                resource_estimate=m["resource"],
                metadata=p["path"]
            )
            path_candidates.append(candidate)
            
        # 3. 设置约束条件
        sla_constraints = SLAConstraints(
            max_latency=constraints.get("max_latency", float('inf')),
            max_memory=constraints.get("max_resource", float('inf')),
            min_accuracy=0.0
        )
        self.sla_optimizer.constraints = sla_constraints
        
        # 4. 执行优化求解
        try:
            best_candidate, final_weights, optimize_info = self.sla_optimizer.optimize(
                path_candidates, adaptive_weights
            )
            
            # 构造返回结果
            result = best_candidate.metadata.copy()
            result["weights_used"] = final_weights
            result["metrics"] = {
                "error": best_candidate.error_estimate,
                "latency": best_candidate.latency_estimate,
                "resource": best_candidate.resource_estimate
            }
            # 附加优化详情（用于反馈学习）
            result["optimize_info"] = optimize_info

            # 如果组合估计误差不优于最佳单模，则回退到单模（防止组合退化）
            if best_single and best_candidate.error_estimate >= best_single["metrics"]["error"] * 0.99:
                single_path = best_single["path"].copy()
                single_path["metrics"] = best_single["metrics"]
                single_path["weights_used"] = {single_path["models"][0]: 1.0}
                single_path["optimize_info"] = {"reason": "fallback_best_single"}
                single_path["strategy"] = "single"
                print(f"      [安全网] 组合未优于单模，回退至 {single_path['models'][0]}")
                return single_path
            
            print(f"      [SLA优化] 选中路径: {result['path_id']} (Weights: {final_weights})")
            return result
            
        except Exception as e:
            print(f"      [SLA优化] 求解失败: {e}，回退到简单加权")
            # 回退逻辑...
            w_error = adaptive_weights["error"]
            w_latency = adaptive_weights["latency"]
            w_resource = adaptive_weights["resource"]
            
            min_error = min(p["metrics"]["error"] for p in evaluated_paths) + 1e-9
            min_latency = min(p["metrics"]["latency"] for p in evaluated_paths) + 1e-9
            min_resource = min(p["metrics"]["resource"] for p in evaluated_paths) + 1e-9
            
            for p in evaluated_paths:
                m = p["metrics"]
                s_error = min_error / (m["error"] + 1e-9)
                s_latency = min_latency / (m["latency"] + 1e-9)
                s_resource = min_resource / (m["resource"] + 1e-9)
                p["score"] = w_error * s_error + w_latency * s_latency + w_resource * s_resource
                
            best_path = max(evaluated_paths, key=lambda x: x["score"])
            result = best_path["path"].copy()
            result["weights_used"] = adaptive_weights
            result["metrics"] = best_path["metrics"]

            # 同样应用单模安全网
            if best_single and best_path["metrics"]["error"] >= best_single["metrics"]["error"] * 0.99:
                single_path = best_single["path"].copy()
                single_path["metrics"] = best_single["metrics"]
                single_path["weights_used"] = {single_path["models"][0]: 1.0}
                single_path["strategy"] = "single"
                single_path["optimize_info"] = {"reason": "fallback_best_single"}
                print(f"      [安全网] 组合未优于单模，回退至 {single_path['models'][0]}")
                return single_path

            return result

    def _generate_candidate_paths(self, available_models: List[str], model_graph: Any = None, 
                                 scenario_id: str = None, available_features: set = None) -> List[Dict[str, Any]]:
        """生成候选模型路径（增强版：集成图推理引擎）
        
        Args:
            available_models: 可用模型列表
            model_graph: 模型知识图谱
            scenario_id: 当前场景ID（用于图推理）
            available_features: 场景实际特征集合（用于图推理）
            
        Returns:
            候选路径列表
        """
        paths = []
        
        # === 0. 基于图谱推理生成路径（优先级最高） ===
        if model_graph and scenario_id and available_features:
            try:
                reasoning_paths = model_graph.infer_optimal_path_by_reasoning(
                    scenario_id, available_features, constraints={'max_models': 3}
                )
                if reasoning_paths:
                    print(f"      [图推理] 找到 {len(reasoning_paths)} 条候选路径,推荐Top-{min(3, len(reasoning_paths))}")
                    for path_id, score in reasoning_paths[:5]:  # 取Top-5
                        # 从图谱中获取路径组成（如果存在）
                        path_composition = model_graph.get_node_attr(path_id, 'composition', [])
                        
                        # 如果图谱中没有composition,尝试从path_id解析
                        if not path_composition:
                            if path_id.startswith('single_'):
                                model_name = path_id.replace('single_', '')
                                if model_name in available_models:
                                    path_composition = [model_name]
                            elif path_id.startswith('complementary_') or path_id.startswith('serial_'):
                                # 从path_id解析模型列表 (例如: complementary_xgboost_reg_lgbm_reg)
                                parts = path_id.split('_', 1)[1] if '_' in path_id else ''
                                potential_models = [m for m in available_models if m in parts]
                                if len(potential_models) >= 2:
                                    path_composition = potential_models[:2]
                        
                        if path_composition and all(m in available_models for m in path_composition):
                            strategy = model_graph.get_node_attr(path_id, 'strategy', 'weighted_mean')
                            if 'single' in path_id:
                                strategy = 'single'
                            elif 'serial' in path_id:
                                strategy = 'stacking'
                            
                            paths.append({
                                "path_id": path_id,
                                "models": path_composition,
                                "weights": {m: 1.0/len(path_composition) for m in path_composition},
                                "strategy": strategy,
                                "_reasoning_score": score  # 标记为推理路径
                            })
                            print(f"        → {path_id} (score={score:.3f}, models={path_composition})")
            except Exception as e:
                print(f"      [图推理] 失败: {e}")
        
        # 1. 单模型路径
        for model in available_models:
            paths.append({
                "path_id": f"single_{model}",
                "models": [model],
                "weights": {model: 1.0},
                "strategy": "single"
            })
            
        # 2. 基于图谱的路径生成
        if model_graph:
            # 查找互补关系 (complementary)
            for model_a in available_models:
                complementary_models = model_graph.neighbors_by_type(model_a, "complementary")
                for model_b in complementary_models:
                    if model_b in available_models:
                        path_id = f"complementary_{model_a}_{model_b}"
                        # 在图谱中实例化路径节点
                        model_graph.instantiate_path(path_id, [model_a, model_b], "weighted_mean")
                        
                        paths.append({
                            "path_id": path_id,
                            "models": [model_a, model_b],
                            "weights": {model_a: 0.6, model_b: 0.4}, # 初始权重
                            "strategy": "weighted_mean"
                        })
                        
            # 查找串行修正关系 (dependency)
            for model_a in available_models:
                dependent_models = model_graph.neighbors_by_type(model_a, "dependency")
                for model_b in dependent_models:
                    if model_b in available_models:
                        path_id = f"serial_{model_a}_{model_b}"
                        model_graph.instantiate_path(path_id, [model_a, model_b], "stacking")
                        
                        paths.append({
                            "path_id": path_id,
                            "models": [model_a, model_b],
                            "weights": {model_a: 0.8, model_b: 0.2},
                            "strategy": "stacking"
                        })

        # 3. 启发式并行组合 (Top 2 & Top 3) - 作为补充
        if len(available_models) >= 2:
            pair = available_models[:2]
            path_id = f"parallel_{'_'.join(pair)}"
            if model_graph:
                model_graph.instantiate_path(path_id, pair, "weighted_mean")
            
            paths.append({
                "path_id": path_id,
                "models": pair,
                "weights": {m: 0.5 for m in pair},
                "strategy": "weighted_mean"
            })
            
        if len(available_models) >= 3:
            triplet = available_models[:3]
            path_id = f"parallel_{'_'.join(triplet)}"
            if model_graph:
                model_graph.instantiate_path(path_id, triplet, "weighted_mean")

            paths.append({
                "path_id": path_id,
                "models": triplet,
                "weights": {m: 0.33 for m in triplet},
                "strategy": "weighted_mean"
            })
            
        return paths

    def _evaluate_path_cost(self, path: Dict[str, Any], scenario_signature: Dict[str, float], 
                           model_graph: Any = None, similar_scenarios: List[Tuple[str, float]] = None) -> Dict[str, float]:
        """预估路径代价 (Error, Latency, Resource) - 融合历史真实性能（优先路径级）"""
        num_models = len(path["models"])
        path_id = path["path_id"]
        
        # === 1. 历史性能回溯（优先查找路径级，再查找模型级） ===
        historical_error = None
        if similar_scenarios and hasattr(self, '_historical_scenarios_data'):
            # 1.1 优先查找完全相同路径的历史表现
            for scenario_id, similarity in similar_scenarios:
                hist_record = next((rec for rec in self._historical_scenarios_data 
                                   if rec[0] == scenario_id), None)
                if hist_record:
                    _, _, performance = hist_record
                    # 检查是否有路径级RMSE
                    if performance.get("path_id") == path_id and "path_rmse" in performance:
                        path_rmse = performance["path_rmse"]
                        # 路径级历史直接使用，加权相似度
                        historical_error = path_rmse * similarity
                        print(f"      使用路径级历史: {path_id} RMSE={path_rmse:.4f} (相似度{similarity:.3f})")
                        break
            
            # 1.2 如果没有路径级，回退到模型级（逐个模型查找）
            if historical_error is None:
                path_models_set = set(path["models"])
                for scenario_id, similarity in similar_scenarios:
                    hist_record = next((rec for rec in self._historical_scenarios_data 
                                       if rec[0] == scenario_id), None)
                    if hist_record:
                        _, _, performance = hist_record
                        # 检查是否有当前路径中模型的真实RMSE
                        for model_id in path["models"]:
                            if model_id in performance:
                                model_rmse = performance[model_id].get('RMSE', None)
                                if model_rmse:
                                    # 加权相似度和真实误差
                                    if historical_error is None:
                                        historical_error = model_rmse * similarity
                                    else:
                                        historical_error += model_rmse * similarity * 0.5  # 累积但衰减
        
        # === 2. 启发式估计（兜底或补充） ===
        base_error = 0.1
        estimated_error = base_error * (0.9 ** (num_models - 1))
        if "complementary" in path["path_id"]:
            estimated_error *= 0.8  # 互补组合有额外收益
        
        # 如果有历史数据，使用加权平均
        if historical_error is not None:
            final_error = 0.7 * historical_error + 0.3 * estimated_error  # 70%历史 + 30%启发式
        else:
            final_error = estimated_error
        
        # ========== 资源估算: 历史观测优先 + 元数据回退 (技术限制) ==========
        # 【当前实现】分级回退策略
        #   1. 优先: 从 historical_scenarios.json 读取真实 _profiling 数据
        #   2. 回退: 使用 model_graph 元数据映射 (low/medium/high)
        #   3. 兜底: 基于模型数量的简单估计
        # 【局限性】
        #   - 元数据粗粒度: low/medium/high 无法反映真实波动
        #   - 无动态建模: 未考虑数据规模/特征维度影响
        #   - 冷启动问题: 新模型初期估算误差大
        # 【改进方向】见 docs/technical_limitations.md 第2节
        # ====================================================================
        
        total_latency_score = 0
        total_resource_score = 0
        
        # 3.1 尝试从历史场景中获取真实性能数据
        historical_profiling = None
        if similar_scenarios and hasattr(self, '_historical_scenarios_data'):
            for scenario_id, similarity in similar_scenarios:
                hist_record = next((rec for rec in self._historical_scenarios_data 
                                   if rec[0] == scenario_id), None)
                if hist_record:
                    _, _, performance = hist_record
                    # 检查是否有profiling数据
                    if '_profiling' in performance:
                        historical_profiling = performance['_profiling']
                        print(f"      [真实性能] 使用历史观测: 耗时={historical_profiling.get('elapsed_seconds', 0)}s")
                        break
        
        # 3.2 使用真实性能数据或回退到资源预测/元数据
        if historical_profiling:
            # 使用真实观测的耗时和内存
            total_latency_score = historical_profiling.get('elapsed_seconds', 1.0) * 1000  # 转为ms
            total_resource_score = historical_profiling.get('memory_mb', 100) / 100  # 归一化
        elif self.enable_resource_prediction and self.resource_predictor and self.resource_predictor.fitted:
            # Phase 2: 使用资源预测模型
            data_size = scenario_signature.get('_data_size', 1000)  # 从签名中获取数据规模
            num_features = scenario_signature.get('_num_features', 30)
            
            for model_id in path["models"]:
                pred = self.resource_predictor.predict_resources(model_id, data_size, num_features)
                total_latency_score += pred['latency_ms']
                total_resource_score += pred['memory_mb'] / 100
            
            print(f"      [资源预测] 延迟: {total_latency_score:.1f}ms, 内存: {total_resource_score*100:.1f}MB")
        elif model_graph:
            # 回退到元数据映射 (low/medium/high)
            for model_id in path["models"]:
                res_cost = model_graph.get_node_attr(model_id, "resource_cost", {})
                
                # 解析 latency
                lat = res_cost.get("latency", "medium")
                if isinstance(lat, (int, float)):
                    total_latency_score += lat
                else:
                    cost_map = {"low": 50, "medium": 100, "high": 200}
                    total_latency_score += cost_map.get(lat, 100)
                
                # 解析 cpu/memory
                cpu = res_cost.get("cpu", "medium")
                cost_map = {"low": 1, "medium": 2, "high": 3}
                total_resource_score += cost_map.get(cpu, 2)
        else:
            total_latency_score = num_models * 100
            total_resource_score = num_models * 2
        
        return {
            "error": final_error,
            "latency": total_latency_score,
            "resource": total_resource_score,
            "used_history": historical_error is not None
        }

    def select_models_by_scenario(self, scenario_signature: Dict[str, float], 
                                available_models: List[str],
                                historical_scenarios: List[Tuple[str, Dict[str, float], Dict[str, float]]] = None,
                                max_models: int = 3) -> List[str]:
        """基于场景特征选择最适合的模型"""
        
        # 基于规则的初步筛选
        rule_based_models = self._rule_based_selection(scenario_signature, available_models)
        
        # 基于历史性能的选择
        if historical_scenarios:
            performance_based_models = self._performance_based_selection(
                scenario_signature, available_models, historical_scenarios
            )
        else:
            performance_based_models = []
            
        # 基于模型特性的选择
        characteristic_based_models = self._characteristic_based_selection(
            scenario_signature, available_models
        )
        
        # 组合所有选择结果
        selected_models = self._combine_selections(
            rule_based_models, performance_based_models, characteristic_based_models, max_models
        )
        
        return selected_models[:max_models]
    
    def _rule_based_selection(self, scenario_signature: Dict[str, float], 
                            available_models: List[str]) -> List[str]:
        """基于规则的模型选择"""
        selected = []
        
        # 获取区域类型
        region_type = scenario_signature.get("region_type", 0.0)
        
        # 住宅区域规则
        if region_type == 1.0:  # residential
            preferred_order = ["prophet", "multimodal_fusion", "lgbm_reg", "xgboost_reg", "arima"]
        # 充电桩区域规则  
        elif region_type == 2.0:  # charging
            preferred_order = ["xgboost_reg", "lgbm_reg", "power_difference", "multimodal_fusion", "prophet"]
        # 服务区规则
        elif region_type == 3.0:  # service_area
            preferred_order = ["prophet", "xgboost_reg", "multimodal_fusion", "power_difference", "arima"]
        else:
            preferred_order = ["lgbm_reg", "xgboost_reg", "prophet", "multimodal_fusion"]
            
        # 根据可用模型筛选
        for model in preferred_order:
            if model in available_models and model not in selected:
                selected.append(model)
                
        return selected
    
    def _performance_based_selection(self, scenario_signature: Dict[str, float],
                                   available_models: List[str],
                                   historical_scenarios: List[Tuple[str, Dict[str, float], Dict[str, float]]]) -> List[str]:
        """基于历史性能的模型选择"""
        # 找到相似的历史场景
        historical_sigs = [(sid, sig) for sid, sig, _ in historical_scenarios]
        similar_scenarios = self.scenario_analyzer.find_similar_scenarios(
            scenario_signature, historical_sigs, top_k=5, similarity_threshold=0.4
        )
        
        if not similar_scenarios:
            return []
            
        # 计算每个模型在相似场景中的平均性能
        model_scores = {}
        for model in available_models:
            scores = []
            for scenario_id, similarity in similar_scenarios:
                # 找到对应的性能数据
                for sid, _, performance in historical_scenarios:
                    if sid == scenario_id and model in performance:
                        # 使用RMSE的倒数作为分数（RMSE越小，分数越高）
                        rmse = performance[model].get('RMSE', float('inf'))
                        if rmse > 0:
                            score = 1.0 / rmse * similarity  # 加权相似度
                            scores.append(score)
                        break
                        
            if scores:
                model_scores[model] = np.mean(scores)
                
        # 按性能排序
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models]
    
    def _characteristic_based_selection(self, scenario_signature: Dict[str, float],
                                      available_models: List[str]) -> List[str]:
        """基于场景特征的模型选择"""
        selected = []
        
        # 高季节性场景偏好Prophet
        seasonal_amplitude = scenario_signature.get("seasonal_amplitude", 0.0)
        if seasonal_amplitude > 0.3 and "prophet" in available_models:
            selected.append("prophet")
            
        # 高变异性场景偏好树模型
        cv_load = scenario_signature.get("cv_load", 0.0)
        if cv_load > 0.2:
            for model in ["xgboost_reg", "lgbm_reg", "catboost_reg"]:
                if model in available_models and model not in selected:
                    selected.append(model)
                    break
                    
        # 强趋势场景偏好ARIMA
        trend_strength = scenario_signature.get("trend_strength", 0.0)
        if trend_strength > 0.5 and "arima" in available_models:
            selected.append("arima")
            
        # 多模态数据场景偏好融合模型
        weather_corr_features = [k for k in scenario_signature.keys() if "corr" in k]
        if len(weather_corr_features) > 2 and "multimodal_fusion" in available_models:
            selected.append("multimodal_fusion")
            
        # 区域差异明显的场景偏好差异分析模型
        peak_valley_ratio = scenario_signature.get("peak_valley_ratio", 1.0)
        if peak_valley_ratio > 2.0 and "power_difference" in available_models:
            selected.append("power_difference")
            
        return selected
    
    def _combine_selections(self, rule_based: List[str], performance_based: List[str],
                          characteristic_based: List[str], max_models: int) -> List[str]:
        """组合不同选择策略的结果"""
        # 使用加权投票机制
        model_votes = {}
        
        # 规则选择权重最高
        for i, model in enumerate(rule_based[:max_models]):
            model_votes[model] = model_votes.get(model, 0) + (max_models - i) * 3
            
        # 性能选择权重中等
        for i, model in enumerate(performance_based[:max_models]):
            model_votes[model] = model_votes.get(model, 0) + (max_models - i) * 2
            
        # 特征选择权重较低
        for i, model in enumerate(characteristic_based[:max_models]):
            model_votes[model] = model_votes.get(model, 0) + (max_models - i) * 1
            
        # 按投票数排序
        sorted_models = sorted(model_votes.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models]
    
    def calculate_model_weights(self, selected_models: List[str], 
                              scenario_signature: Dict[str, float],
                              historical_performance: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
        """计算模型权重"""
        if not selected_models:
            return {}
            
        if len(selected_models) == 1:
            return {selected_models[0]: 1.0}
            
        weights = {}
        
        # 基础权重分配
        base_weight = 1.0 / len(selected_models)
        for model in selected_models:
            weights[model] = base_weight
            
        # 根据历史性能调整权重
        if historical_performance:
            total_performance = 0.0
            model_performances = {}
            
            for model in selected_models:
                if model in historical_performance:
                    # 使用RMSE的倒数作为性能指标
                    rmse = historical_performance[model].get('RMSE', float('inf'))
                    if rmse > 0:
                        performance = 1.0 / rmse
                        model_performances[model] = performance
                        total_performance += performance
                        
            # 重新分配权重
            if total_performance > 0:
                for model in selected_models:
                    if model in model_performances:
                        weights[model] = model_performances[model] / total_performance
                    else:
                        weights[model] = 0.1  # 给未知性能的模型较小权重
                        
        # 根据场景特征微调权重
        weights = self._adjust_weights_by_scenario(weights, scenario_signature)
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
            
        return weights
    
    def _adjust_weights_by_scenario(self, weights: Dict[str, float], 
                                  scenario_signature: Dict[str, float]) -> Dict[str, float]:
        """根据场景特征调整权重"""
        adjusted_weights = weights.copy()
        
        # 高季节性场景提升Prophet权重
        seasonal_amplitude = scenario_signature.get("seasonal_amplitude", 0.0)
        if seasonal_amplitude > 0.3 and "prophet" in adjusted_weights:
            adjusted_weights["prophet"] *= 1.2
            
        # 高变异性场景提升树模型权重
        cv_load = scenario_signature.get("cv_load", 0.0)
        if cv_load > 0.2:
            for model in ["xgboost_reg", "lgbm_reg", "catboost_reg"]:
                if model in adjusted_weights:
                    adjusted_weights[model] *= 1.1
                    
        # 强天气相关性提升融合模型权重
        weather_corr_features = [k for k in scenario_signature.keys() if "corr" in k]
        avg_weather_corr = np.mean([scenario_signature[k] for k in weather_corr_features]) if weather_corr_features else 0.0
        if avg_weather_corr > 0.3 and "multimodal_fusion" in adjusted_weights:
            adjusted_weights["multimodal_fusion"] *= 1.15
            
        return adjusted_weights
    
    def create_ensemble(self, selected_models: List[str], weights: Dict[str, float],
                       ensemble_method: str = "weighted") -> Any:
        """创建集成模型"""
        if ensemble_method == "weighted":
            return WeightedBlender(weights=weights)
        elif ensemble_method == "stacking":
            return StackingBlender()
        else:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")


# 保持向后兼容的函数
def choose_models_by_rules(region_type: str, rules: List[dict]) -> List[str]:
    """向后兼容的规则选择函数"""
    for r in rules:
        filt = r.get("scenario_filter", {})
        if filt.get("region_type") == region_type:
            return r.get("prefer", []) or r.get("fallback", [])
    return []


def build_weighted_blender(weights: Dict[str, float]) -> WeightedBlender:
    """向后兼容的加权混合器构建函数"""
    return WeightedBlender(weights=weights)
