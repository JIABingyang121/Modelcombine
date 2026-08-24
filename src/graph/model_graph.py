from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import networkx as nx
import numpy as np
import time
from .hyperpath import (
    add_scenario_hyperpath_edge,
    apply_hyperpath_temporal_update,
    get_hyperpaths_for_model,
    get_top_hyperpaths_for_scenario,
    instantiate_hyperpath,
    update_hyperpath_metrics,
)
from .temporal_relations import HawkesRelationUpdater, RelationEvent


class ModelGraph:
    """
    模型资产知识图谱
    Ontology: Scenario -[has_feature]-> Feature -[input_to]-> Model -[part_of]-> Path -[produces]-> Performance
    """
    def __init__(self):
        self.G = nx.DiGraph()

    # --- Node Management ---

    def add_scenario_node(self, scenario_id: str, attributes: Dict[str, Any]):
        """添加场景节点"""
        self.G.add_node(scenario_id, node_type="scenario", **attributes)

    def add_feature_node(self, feature_id: str, description: str = ""):
        """添加特征节点"""
        self.G.add_node(feature_id, node_type="feature", description=description)

    def add_model_node(self, model_id: str, metadata: Dict[str, Any]):
        """添加模型节点"""
        self.G.add_node(model_id, node_type="model", **metadata)

    def add_path_node(
        self,
        path_id: str,
        composition: List[str],
        strategy: str,
        **attrs: Any,
    ):
        """添加路径节点 (模型组合)"""
        instantiate_hyperpath(
            self,
            path_id=path_id,
            models=composition,
            strategy=strategy,
            created_from=attrs.pop("created_from", ""),
            ordered=attrs.pop("ordered", None),
            metrics=attrs.pop("metrics", None),
            evidence_ref=attrs.pop("evidence_ref", ""),
        )
        self.G.nodes[path_id].update(attrs)

    # --- Edge Management ---

    def add_scenario_feature_edge(self, scenario_id: str, feature_id: str, weight: float = 1.0):
        """场景 -> 特征"""
        self.G.add_edge(scenario_id, feature_id, edge_type="has_feature", weight=weight)

    def add_feature_model_edge(self, feature_id: str, model_id: str, required: bool = True):
        """特征 -> 模型"""
        self.G.add_edge(feature_id, model_id, edge_type="input_to", required=required)

    def add_model_path_edge(self, model_id: str, path_id: str, order: int):
        """模型 -> 路径"""
        self.G.add_edge(model_id, path_id, edge_type="part_of", order=order)

    def add_scenario_path_edge(
        self,
        scenario_id: str,
        path_id: str,
        performance_score: float,
        *,
        relation_type: str = "recommended_for",
        evidence_ref: str = "",
        **attrs: Any,
    ):
        """场景 -> 路径 (历史推荐)"""
        add_scenario_hyperpath_edge(
            self,
            scenario_id,
            path_id,
            relation_type=relation_type,
            weight=performance_score,
            evidence_ref=evidence_ref,
            **attrs,
        )

    def add_scenario_model_edge(
        self,
        scenario_id: str,
        model_id: str,
        weight: float = 1.0,
        *,
        relation_type: str = "recommended_for",
    ):
        """场景 -> 模型 (直接推荐，不要求 model_id 是 Path 超边)"""
        self.G.add_edge(scenario_id, model_id, edge_type=relation_type, weight=weight)

    # --- Dynamic Updates ---

    def update_edge_weight(self, source: str, target: str, feedback_score: float, learning_rate: float = 0.1):
        """基于反馈更新边权重 (增量学习)"""
        if self.G.has_edge(source, target):
            current_weight = self.G[source][target].get("weight", 1.0)
            # 简单的移动平均更新
            new_weight = (1 - learning_rate) * current_weight + learning_rate * feedback_score
            self.G[source][target]["weight"] = new_weight
            self.G[source][target]["last_updated"] = time.time()

    def update_temporal_relation_strength(
        self,
        events: List[RelationEvent],
        *,
        now: Any,
        updater: Optional[HawkesRelationUpdater] = None,
        create_missing: bool = False,
    ) -> Dict[str, Any]:
        """基于带时间衰减的关系事件更新动态图谱边强度。"""
        updater = updater or HawkesRelationUpdater()
        return updater.apply_to_graph(
            self,
            events,
            now=now,
            create_missing=create_missing,
        )

    def apply_hyperpath_temporal_update(
        self,
        events: List[RelationEvent],
        *,
        now: Any,
        updater: Optional[HawkesRelationUpdater] = None,
        create_missing: bool = False,
    ) -> Dict[str, Any]:
        """更新场景到 Path 超边及成员模型到 Path 的时序关系强度。"""
        return apply_hyperpath_temporal_update(
            self,
            events,
            now=now,
            updater=updater,
            create_missing=create_missing,
        )

    # --- Query & Reasoning ---

    def get_candidate_paths_for_scenario(self, scenario_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """根据场景获取推荐路径"""
        if not self.G.has_node(scenario_id):
            return []

        # 查找直接连接的推荐路径
        recommendations = []
        for _, target, data in self.G.out_edges(scenario_id, data=True):
            if data.get("edge_type") == "recommended_for":
                recommendations.append((target, data.get("weight", 0.0)))

        # 按权重排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

    def get_models_by_feature(self, feature_id: str) -> List[str]:
        """查找使用特定特征的所有模型"""
        models = []
        for _, target, data in self.G.out_edges(feature_id, data=True):
            if data.get("edge_type") == "input_to":
                models.append(target)
        return models

    def infer_optimal_path_by_reasoning(self, scenario_id: str, available_features: set,
                                       constraints: Dict[str, float] = None,
                                       return_details: bool = False):
        """基于图谱多跳推理生成候选路径（显式推理链）

        推理链: Scenario -> Feature -> Model -> Path -> Historical Performance

        Args:
            scenario_id: 场景ID
            available_features: 场景实际具备的特征集合
            constraints: 约束条件

        Returns:
            List[(path_id, score)]: 推荐路径及得分，按得分降序
        """
        constraints = constraints or {}

        # Step 1: 从可用特征出发，找到所有可以使用这些特征的模型
        candidate_models = set()
        for feat in sorted(available_features):
            if self.G.has_node(feat):
                for _, model, data in self.G.out_edges(feat, data=True):
                    if data.get('edge_type') == 'input_to':
                        # 检查模型是否所有required特征都满足
                        model_required = self.get_node_attr(model, 'input_constraints', {}).get('features', [])
                        if set(model_required).issubset(available_features):
                            candidate_models.add(model)

        if not candidate_models:
            return []

        # Step 2: 查找这些模型参与的所有路径（通过 part_of 边）+ 动态生成单模型路径
        candidate_paths = {}

        # 2.1 查找图谱中已存在的路径
        for model in sorted(candidate_models):
            for _, path, data in self.G.out_edges(model, data=True):
                if data.get('edge_type') == 'part_of':
                    if path not in candidate_paths:
                        candidate_paths[path] = []
                    candidate_paths[path].append(model)

        # 2.2 动态生成单模型路径（解决冷启动问题）
        for model in sorted(candidate_models):
            single_path_id = f"single_{model}"
            if single_path_id not in candidate_paths:
                # 注册到图谱 (确保节点存在)
                self.instantiate_path(single_path_id, [model], "single")
                candidate_paths[single_path_id] = [model]

        # 2.3 基于图谱关系动态生成组合路径
        # 先收集、后实例化：instantiate_path 会给 model_a 新增 part_of 出边，
        # 若边遍历边实例化会触发 NetworkX 的 "dictionary changed size during
        # iteration"（历史上被调用方 try/except 静默吞掉，导致 reasoning 失效）。
        generated_paths = []
        for model_a in sorted(candidate_models):
            for _, target, data in self.G.out_edges(model_a, data=True):
                if data.get('edge_type') == 'complementary' and target in candidate_models:
                    generated_paths.append(
                        (f"complementary_{model_a}_{target}", [model_a, target], "weighted_mean")
                    )
                elif data.get('edge_type') == 'dependency' and target in candidate_models:
                    generated_paths.append(
                        (f"serial_{model_a}_{target}", [model_a, target], "stacking")
                    )
        for path_id, members, strategy in generated_paths:
            if path_id not in candidate_paths:
                self.instantiate_path(path_id, members, strategy)
                candidate_paths[path_id] = members

        # Step 3: 为每条路径计算历史性能得分
        path_scores = []
        for path_id, path_models in candidate_paths.items():
            # 3.1 获取路径在相似场景下的历史得分
            hist_score = self._get_path_historical_score(path_id, scenario_id)

            # 3.2 检查约束（简化版：基于路径模型数量）
            if constraints.get('max_models') and len(path_models) > constraints['max_models']:
                continue

            path_scores.append((path_id, hist_score))

        # Step 4: 按得分排序返回
        # 冷启动路径没有历史分差时，必须用路径 ID 作为稳定次级键。
        # 否则 candidate_models 的 set 遍历顺序会让 hybrid reasoning 在不同
        # Python 进程中选到不同首路径，进而改写最终模型集合。
        path_scores.sort(key=lambda item: (-item[1], item[0]))

        if return_details:
            from .path_reasoning import PathReasoningScorer

            scorer = PathReasoningScorer()
            details = [
                scorer.score(self, scenario_id, path_id, available_features=set(available_features))
                for path_id, _ in path_scores
            ]
            details.sort(key=lambda item: item.final_score, reverse=True)
            return details

        return path_scores

    def _get_path_historical_score(self, path_id: str, scenario_id: str) -> float:
        """查询路径在相似场景下的历史得分

        Args:
            path_id: 路径ID
            scenario_id: 当前场景ID

        Returns:
            平均历史得分（0-1之间，越高越好）
        """
        import numpy as np

        # 1. 查找当前场景是否有直接的推荐记录
        if self.G.has_edge(scenario_id, path_id):
            edge_data = self.G[scenario_id][path_id]
            if edge_data.get('edge_type') == 'recommended_for':
                return edge_data.get('weight', 0.0)

        # 2. 查找相似场景的推荐记录
        similar_scenarios = self._find_similar_scenario_nodes(scenario_id)
        scores = []
        for sim_scenario in similar_scenarios:
            if self.G.has_edge(sim_scenario, path_id):
                edge_data = self.G[sim_scenario][path_id]
                if edge_data.get('edge_type') == 'recommended_for':
                    weight = edge_data.get('weight', 0.0)
                    scores.append(weight)

        # 3. 返回平均得分（如果没有历史，返回中性得分0.5）
        return np.mean(scores) if scores else 0.5

    def _find_similar_scenario_nodes(self, scenario_id: str, max_similar: int = 5) -> List[str]:
        """查找相似场景节点（基于签名相似度+region_type）

        Args:
            scenario_id: 当前场景ID
            max_similar: 最大返回数量

        Returns:
            相似场景ID列表（按相似度降序）
        """
        if not self.G.has_node(scenario_id):
            return []

        current_region = self.G.nodes[scenario_id].get('region', '')
        current_type = self.G.nodes[scenario_id].get('region_type', '')
        current_sig = self.G.nodes[scenario_id].get('signature', {})

        # 计算场景相似度
        scenario_scores = []
        for node_id in self.G.nodes():
            if node_id == scenario_id:
                continue

            node_data = self.G.nodes[node_id]
            if node_data.get('node_type') != 'scenario':
                continue

            # 计算相似度得分
            similarity_score = 0.0

            # 1. 相同区域类型 +0.5
            if node_data.get('region_type') == current_type:
                similarity_score += 0.5

            # 2. 签名相似度（基于特征值的欧氏距离）+0.5
            if current_sig and node_data.get('signature'):
                node_sig = node_data['signature']
                common_keys = set(current_sig.keys()) & set(node_sig.keys())
                if common_keys:
                    # 计算归一化欧氏距离
                    distances = []
                    for key in common_keys:
                        if isinstance(current_sig[key], (int, float)) and isinstance(node_sig[key], (int, float)):
                            distances.append(abs(current_sig[key] - node_sig[key]))

                    if distances:
                        avg_distance = np.mean(distances)
                        # 转换为相似度 (距离越小相似度越高)
                        sig_similarity = 1.0 / (1.0 + avg_distance)
                        similarity_score += 0.5 * sig_similarity

            if similarity_score > 0:
                scenario_scores.append((node_id, similarity_score))

        # 按相似度降序排序
        scenario_scores.sort(key=lambda x: x[1], reverse=True)

        return [sid for sid, _ in scenario_scores[:max_similar]]

    def instantiate_path(
        self,
        path_id: str,
        models: List[str],
        strategy: str = "weighted_mean",
        *,
        created_from: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        evidence_ref: str = "",
    ) -> str:
        """实例化路径节点，如果不存在则创建"""
        return instantiate_hyperpath(
            self,
            path_id=path_id,
            models=models,
            strategy=strategy,
            created_from=created_from,
            metrics=metrics,
            evidence_ref=evidence_ref,
        )

    def update_path_metrics(
        self,
        path_id: str,
        metrics: Dict[str, Any],
        *,
        evidence_ref: str = "",
    ) -> Dict[str, Any]:
        """更新 Path 超边组合级指标。"""
        return update_hyperpath_metrics(
            self,
            path_id,
            metrics,
            evidence_ref=evidence_ref,
        )

    def get_top_hyperpaths_for_scenario(
        self,
        scenario_id: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """查询场景关联的 top-k Path 超边。"""
        return get_top_hyperpaths_for_scenario(
            self,
            scenario_id,
            top_k=top_k,
        )

    def get_hyperpaths_for_model(
        self,
        model_id: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """查询模型参与过的 Path 超边。"""
        return get_hyperpaths_for_model(
            self,
            model_id,
            top_k=top_k,
        )

    def get_node_attr(self, node_id: str, attr: str, default: Any = None) -> Any:
        """获取节点属性"""
        if self.G.has_node(node_id):
            return self.G.nodes[node_id].get(attr, default)
        return default

    # --- Backward Compatibility ---

    def add_model(self, model_id: str, **attrs):
        """兼容旧接口"""
        self.add_model_node(model_id, attrs)

    def add_relation(self, source: str, target: str, rel_type: str, weight: float = 1.0):
        """兼容旧接口"""
        self.G.add_edge(source, target, edge_type=rel_type, weight=weight)

    def neighbors_by_type(self, model_id: str, rel_type: str) -> List[str]:
        """兼容旧接口"""
        out = []
        for _, tgt, data in self.G.out_edges(model_id, data=True):
            if data.get("edge_type") == rel_type:
                out.append(tgt)
        return out

    def find_paths(self, start: str, end: str, max_len: int = 3) -> List[List[str]]:
        """兼容旧接口"""
        paths = []
        try:
            for path in nx.all_simple_paths(self.G, start, end, cutoff=max_len):
                paths.append(path)
        except nx.NodeNotFound:
            pass
        return paths

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [{"id": n, **self.G.nodes[n]} for n in self.G.nodes],
            "edges": [{"source": s, "target": t, **d} for s, t, d in self.G.edges(data=True)],
        }

    # --- Persistence ---

    def save_graph(self, path: str):
        """保存图谱结构和边权到文件（跨进程持久化）

        Args:
            path: 保存路径（建议 .pkl 后缀）
        """
        import pickle
        graph_data = {
            "nodes": dict(self.G.nodes(data=True)),
            "edges": list(self.G.edges(data=True))
        }
        with open(path, 'wb') as f:
            pickle.dump(graph_data, f)
        print(f"图谱已保存到: {path}")

    def load_graph(self, path: str):
        """从文件加载持久化的图谱（恢复边权和节点属性）

        Args:
            path: 图谱文件路径
        """
        import pickle
        if not os.path.exists(path):
            print(f"图谱文件不存在，跳过加载: {path}")
            return

        with open(path, 'rb') as f:
            graph_data = pickle.load(f)

        # 恢复节点
        for node_id, attrs in graph_data["nodes"].items():
            self.G.add_node(node_id, **attrs)

        # 恢复边
        for source, target, attrs in graph_data["edges"]:
            self.G.add_edge(source, target, **attrs)

        print(f"图谱已从 {path} 加载，节点数: {self.G.number_of_nodes()}, 边数: {self.G.number_of_edges()}")
