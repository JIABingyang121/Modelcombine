from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Sequence
from datetime import datetime
from ..utils.io import load_yaml, ensure_dir, save_json, load_json
from ..features.build_features import add_time_features, add_holiday_feature, join_weather, add_lag_rolling, build_matrix
from ..models.registry import model_registry, load_asset_yaml
from ..graph.model_graph import ModelGraph
from ..core.enums import ModelLifecycleStage, TaskType
from ..core.schema import DataContract, ModelManifest, ScenarioDefinition
from ..core.scenario_id import compute_scenario_id
from ..core.solver import build_solver
from ..core.solver.context import SolveContext
from ..selector.scenario_similarity import PowerScenarioAnalyzer
from ..selector.combinator import PowerModelCombinator
from ..utils.metrics import evaluate_all
from ..features.event_features import EventFeatureExtractor
from ..features.scene_encoder import ScenarioEncoder


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


from .prediction_pool import build_region_prediction_bundle
from .protocol_b_adapter import DemoProtocolBAdapter, build_shadow_comparison


def load_configs() -> Dict[str, Any]:
    assets = load_asset_yaml(os.path.join(PROJECT_ROOT, "configs", "model_assets.yaml"))
    pipe = load_yaml(os.path.join(PROJECT_ROOT, "configs", "pipeline.yaml"))
    return {"assets": assets, "pipeline": pipe}


def _graph_state_path() -> str:
    """图谱持久化路径；MODELCOMBINE_GRAPH_STATE_PATH 可覆盖（三策略对照实验按组隔离用）。"""
    return os.environ.get("MODELCOMBINE_GRAPH_STATE_PATH") or os.path.join(
        PROJECT_ROOT, "reports", "graph_state.pkl"
    )


def _edge_update_strategy() -> str:
    """边权更新策略：moving_average（默认，EMA）| fixed（不更新）| hawkes（仅时序衰减）。"""
    value = os.environ.get("MODELCOMBINE_PIPELINE_EDGE_UPDATE_STRATEGY", "").strip().lower()
    return value or "moving_average"


# 三态决策后端（System A/B 合一 Task 3）。迁移期默认仍是旧 combinator。
BACKEND_COMBINATOR = "combinator"
BACKEND_PROTOCOL_B_SHADOW = "protocol_b_shadow"
BACKEND_PROTOCOL_B = "protocol_b"
_VALID_BACKEND_MODES = (BACKEND_COMBINATOR, BACKEND_PROTOCOL_B_SHADOW, BACKEND_PROTOCOL_B)


def resolve_backend_mode(value: Optional[str] = None) -> str:
    """解析 `MODELCOMBINE_PIPELINE_BACKEND`。

    - `protocol_b`（**默认，Task 8 起**）：System B 为唯一决策主干。
    - `protocol_b_shadow`：两套都跑，最终输出仍取 combinator（审计用）。
    - `combinator`：**迁移期显式回退**，必须由用户主动设置，不再是隐式默认。

    未知值直接报错——静默回退会让"以为在跑 Protocol B、实际跑的是旧引擎"这类
    误判混进实验记录，这正是本计划要终结的问题。
    """
    raw = value if value is not None else os.environ.get("MODELCOMBINE_PIPELINE_BACKEND", "")
    mode = (raw or "").strip().lower() or BACKEND_PROTOCOL_B
    if mode not in _VALID_BACKEND_MODES:
        raise ValueError(
            f"MODELCOMBINE_PIPELINE_BACKEND={raw!r} is not a valid backend mode; "
            f"expected one of {list(_VALID_BACKEND_MODES)} (no silent fallback)"
        )
    return mode


class PowerPredictionPipeline:
    """电力需求预测流水线"""

    @property
    def backend_mode(self) -> str:
        """当前决策后端（读环境变量，未知值报错）。见 resolve_backend_mode。"""
        return resolve_backend_mode()

    def __init__(self, config_path: str = None, enable_phase2: bool = True, enable_phase3: bool = True):
        self.config = self.load_configs(config_path)
        self.scenario_analyzer = PowerScenarioAnalyzer()
        
        # Phase 2: 初始化轻量级学习组件
        self.enable_phase2 = enable_phase2
        self.model_combinator = PowerModelCombinator(
            enable_adaptive_weights=enable_phase2,
            enable_resource_prediction=enable_phase2
        )
        
        # 加载历史场景库
        self.history_path = os.path.join(PROJECT_ROOT, "reports", "historical_scenarios.json")
        self.historical_scenarios = self._load_history()
        
        # Phase 2: 加载持久化的学习模型
        if enable_phase2:
            self._load_phase2_models()
        
        # Phase 3: 初始化高级特征和优化组件
        self.enable_phase3 = enable_phase3
        if enable_phase3:
            self.event_extractor = EventFeatureExtractor()
            self.scenario_encoder = ScenarioEncoder(encoding_dim=16)
            self._load_phase3_models()
        
    def _load_history(self) -> List[Tuple[str, Dict[str, float], Dict[str, float]]]:
        """加载历史场景数据"""
        data = load_json(self.history_path)
        if isinstance(data, list):
            # 转换回 Tuple 格式
            return [(item["id"], item["signature"], item["performance"]) for item in data]
        return []

    def _save_history(self):
        """保存历史场景数据"""
        # 转换为可序列化格式
        data = [
            {"id": sid, "signature": sig, "performance": perf}
            for sid, sig, perf in self.historical_scenarios
        ]
        save_json(data, self.history_path)
    
    def _load_phase2_models(self):
        """加载 Phase 2 持久化模型"""
        models_dir = os.path.join(PROJECT_ROOT, "reports", "phase2_models")
        
        # 加载自适应权重历史
        if self.model_combinator.weight_manager:
            weight_history_path = os.path.join(models_dir, "weight_history.json")
            if os.path.exists(weight_history_path):
                try:
                    history_data = load_json(weight_history_path)
                    self.model_combinator.weight_manager.weight_history = [
                        (h["signature"], tuple(h["weights"]), tuple(h["performance"]))
                        for h in history_data
                    ]
                    print(f"[Phase 2] 已加载 {len(self.model_combinator.weight_manager.weight_history)} 条权重历史")
                except Exception as e:
                    print(f"[Phase 2] 加载权重历史失败: {e}")
        
        # 加载资源预测模型
        if self.model_combinator.resource_predictor:
            try:
                self.model_combinator.resource_predictor.load_models(models_dir)
                print(f"[Phase 2] 已加载资源预测模型")
            except Exception as e:
                print(f"[Phase 2] 加载资源预测模型失败: {e}")
    
    def _save_phase2_models(self):
        """保存 Phase 2 持久化模型"""
        models_dir = os.path.join(PROJECT_ROOT, "reports", "phase2_models")
        ensure_dir(models_dir)
        
        # 保存自适应权重历史
        if self.model_combinator.weight_manager:
            weight_history_path = os.path.join(models_dir, "weight_history.json")
            try:
                history_data = [
                    {
                        "signature": h[0] if isinstance(h[0], dict) else dict(zip(["feat_" + str(i) for i in range(len(h[0]))], h[0])),
                        "weights": list(h[1]),
                        "performance": list(h[2])
                    }
                    for h in self.model_combinator.weight_manager.weight_history
                ]
                save_json(history_data, weight_history_path)
                
                # 计算权重统计信息
                n_samples = len(history_data)
                if n_samples > 0:
                    weights_array = np.array([h["weights"] for h in history_data])
                    weights_mean = weights_array.mean(axis=0)
                    weights_std = weights_array.std(axis=0)
                    print(f"[Phase 2] 已保存 {n_samples} 条权重历史")
                    print(f"  权重均值: Error={weights_mean[0]:.3f}, Latency={weights_mean[1]:.3f}, Resource={weights_mean[2]:.3f}")
                    print(f"  权重标准差: Error={weights_std[0]:.3f}, Latency={weights_std[1]:.3f}, Resource={weights_std[2]:.3f}")
                    
                    # 判断是否收敛（标准差 < 0.05 表示收敛）
                    if n_samples >= 10 and np.all(weights_std < 0.05):
                        print(f"  [收敛状态] 权重已稳定，标准差 < 0.05")
                    elif n_samples >= 10:
                        print(f"  [学习中] 权重仍在调整，需要更多样本")
            except Exception as e:
                print(f"[Phase 2] 保存权重历史失败: {e}")
        
        # 保存资源预测模型
        if self.model_combinator.resource_predictor:
            try:
                predictor = self.model_combinator.resource_predictor
                predictor.save_models(models_dir)
                
                # 输出资源预测器状态
                n_samples = len(predictor.training_data)
                if predictor.fitted:
                    print(f"[Phase 2] 已保存资源预测模型（样本数: {n_samples}, 已训练）")
                else:
                    print(f"[Phase 2] 资源预测器样本数: {n_samples}/5（未达到训练阈值）")
            except Exception as e:
                print(f"[Phase 2] 保存资源预测模型失败: {e}")
    
    def _load_phase3_models(self):
        """加载 Phase 3 持久化模型"""
        models_dir = os.path.join(PROJECT_ROOT, "reports", "phase3_models")
        
        # 加载场景编码器
        encoder_path = os.path.join(models_dir, "scenario_encoder.json")
        if os.path.exists(encoder_path):
            try:
                self.scenario_encoder.load(encoder_path)
                encoder_info = self.scenario_encoder.get_encoding_info()
                print(f"[Phase 3] 已加载场景编码器（维度: {encoder_info['encoding_dim']}, "
                      f"特征数: {encoder_info['n_features']}, 已训练: {encoder_info['is_fitted']}）")
            except Exception as e:
                print(f"[Phase 3] 加载场景编码器失败: {e}")
    
    def _save_phase3_models(self):
        """保存 Phase 3 持久化模型"""
        models_dir = os.path.join(PROJECT_ROOT, "reports", "phase3_models")
        ensure_dir(models_dir)
        
        # 保存场景编码器
        if self.scenario_encoder.is_fitted:
            encoder_path = os.path.join(models_dir, "scenario_encoder.json")
            try:
                self.scenario_encoder.save(encoder_path)
                encoder_info = self.scenario_encoder.get_encoding_info()
                print(f"[Phase 3] 已保存场景编码器（维度: {encoder_info['encoding_dim']}, "
                      f"特征数: {encoder_info['n_features']}）")
            except Exception as e:
                print(f"[Phase 3] 保存场景编码器失败: {e}")
        
    def load_configs(self, config_path: str = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path:
            assets = load_asset_yaml(config_path)
            pipe = load_yaml(config_path.replace("model_assets", "pipeline"))
        else:
            assets = load_asset_yaml(os.path.join(PROJECT_ROOT, "configs", "model_assets.yaml"))
            pipe = load_yaml(os.path.join(PROJECT_ROOT, "configs", "pipeline.yaml"))
        return {"assets": assets, "pipeline": pipe}
    
    def ensure_power_data(self) -> None:
        """确保电力数据存在（不再自动生成模拟数据）。"""
        pipe_cfg = self.config["pipeline"]
        root = os.path.join(PROJECT_ROOT, pipe_cfg["data"]["root"].replace("/", os.sep))
        
        if not os.path.exists(os.path.join(root, "load.csv")):
            raise FileNotFoundError(
                f"未找到数据文件: {os.path.join(root, 'load.csv')}。"
                "当前流程仅支持真实数据，请先完成数据准备。"
            )
            
    def build_power_features(self) -> pd.DataFrame:
        """构建电力预测特征"""
        pipe_cfg = self.config["pipeline"]
        root = os.path.join(PROJECT_ROOT, pipe_cfg["data"]["root"].replace("/", os.sep))
        
        # 加载数据
        load_df = pd.read_csv(os.path.join(root, "load.csv"))
        weather_df = pd.read_csv(os.path.join(root, "weather.csv"))
        
        # 尝试加载节假日数据
        holiday_path = os.path.join(root, "holiday.csv")
        if os.path.exists(holiday_path):
            holiday_df = pd.read_csv(holiday_path)
        else:
            holiday_df = None
            
        # 确保时间戳格式一致
        load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        if holiday_df is not None:
            holiday_df['timestamp'] = pd.to_datetime(holiday_df['timestamp'])
        
        # 合并数据
        df = join_weather(load_df, weather_df)
        
        # 添加时间特征
        df = add_time_features(df)
        
        # 添加节假日特征
        if holiday_df is not None:
            df = df.merge(holiday_df, on="timestamp", how="left")
        else:
            df = add_holiday_feature(df, country="CN")
            
        # 添加滞后和滚动特征
        df = add_lag_rolling(df, 
                           lags=pipe_cfg["features"]["lags"], 
                           rolling_cfg=pipe_cfg["features"]["rolling"])
        
        # 添加电力特定特征
        df = self._add_power_specific_features(df)
        
        # 数据清理和类型转换
        df = self._clean_and_convert_data(df)
        
        # 清理数据
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def _add_power_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加电力特定特征"""
        df = df.copy()
        
        # 添加峰谷时段标记
        df['is_peak_hour'] = df['hour'].isin([7, 8, 18, 19, 20]).astype(int)
        df['is_valley_hour'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
        
        # 添加工作时间标记
        df['is_work_hour'] = df['hour'].between(9, 17).astype(int)
        
        # 添加季节特征
        df['season'] = ((df['month'] - 1) // 3 + 1)  # 1-春, 2-夏, 3-秋, 4-冬
        df['is_summer'] = (df['season'] == 2).astype(int)
        df['is_winter'] = (df['season'] == 4).astype(int)
        
        # 添加温度舒适度特征
        if 'temp' in df.columns:
            df['temp_comfort'] = 1.0 - np.abs(df['temp'] - 22) / 30  # 22度最舒适
            df['temp_comfort'] = np.clip(df['temp_comfort'], 0, 1)
            
            # 添加制冷/供暖需求指标
            df['cooling_demand'] = np.maximum(0, df['temp'] - 26)
            df['heating_demand'] = np.maximum(0, 18 - df['temp'])
            
        # 添加湿度舒适度
        if 'humidity' in df.columns:
            df['humidity_comfort'] = 1.0 - np.abs(df['humidity'] - 50) / 50  # 50%最舒适
            df['humidity_comfort'] = np.clip(df['humidity_comfort'], 0, 1)
            
        # 添加区域类型编码
        if 'region_type' in df.columns:
            region_type_map = {'residential': 1, 'charging': 2, 'service_area': 3}
            df['region_type_encoded'] = df['region_type'].map(region_type_map).fillna(0)
        elif 'region' in df.columns:
            # 从区域名称推断类型
            df['region_type_encoded'] = df['region'].apply(
                lambda x: 1 if 'residential' in x.lower() else 
                         (2 if 'charging' in x.lower() else 3)
            )
            
        return df
    
    def _clean_and_convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理和转换数据类型"""
        df = df.copy()
        
        # 处理重复的is_weekend列
        if 'is_weekend_x' in df.columns and 'is_weekend_y' in df.columns:
            df['is_weekend'] = df['is_weekend_x'].fillna(df['is_weekend_y'])
            df = df.drop(['is_weekend_x', 'is_weekend_y'], axis=1)
        elif 'is_weekend_x' in df.columns:
            df['is_weekend'] = df['is_weekend_x']
            df = df.drop(['is_weekend_x'], axis=1)
        elif 'is_weekend_y' in df.columns:
            df['is_weekend'] = df['is_weekend_y']
            df = df.drop(['is_weekend_y'], axis=1)
            
        # 转换is_workday为数值类型
        if 'is_workday' in df.columns:
            # 处理字符串形式的布尔值
            if df['is_workday'].dtype == 'object':
                df['is_workday'] = df['is_workday'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0)
            else:
                df['is_workday'] = df['is_workday'].astype(int)
            
        # 确保所有布尔列都是数值类型
        bool_columns = df.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            df[col] = df[col].astype(int)
            
        # 移除数据量过少的区域
        region_counts = df['region'].value_counts()
        min_samples = 50  # 最少样本数
        valid_regions = region_counts[region_counts >= min_samples].index
        
        if len(valid_regions) < len(region_counts):
            removed_regions = region_counts[region_counts < min_samples].index.tolist()
            print(f"移除数据量不足的区域: {removed_regions}")
            df = df[df['region'].isin(valid_regions)]
            
        return df

    def split_train_test(self, df: pd.DataFrame, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割训练测试集"""
        df = df.sort_values(["region", "timestamp"]).copy()
        last_ts = pd.to_datetime(df["timestamp"]).max()
        cutoff = last_ts - pd.Timedelta(days=test_days)
        train = df[pd.to_datetime(df["timestamp"]) <= cutoff]
        test = df[pd.to_datetime(df["timestamp"]) > cutoff]
        return train, test

    def build_model_graph(self) -> ModelGraph:
        """构建模型关系图谱（支持从持久化文件加载）"""
        assets_cfg = self.config["assets"]
        mg = ModelGraph()
        
        # 尝试加载持久化的图谱（恢复边权）
        graph_persist_path = _graph_state_path()
        mg.load_graph(graph_persist_path)
        
        # 1. 添加模型节点和特征节点
        for m in assets_cfg.get("models", []):
            mg.add_model_node(m["id"], metadata=m)
            
            # 解析并添加特征节点
            input_constraints = m.get("input_constraints", {})
            features = input_constraints.get("features", [])
            for feat in features:
                if not mg.G.has_node(feat):
                    mg.add_feature_node(feat, description=f"Feature: {feat}")
                # 连接特征 -> 模型
                mg.add_feature_model_edge(feat, m["id"], required=True)
            
        # 2. 添加模型关系边
        for r in assets_cfg.get("relations", []):
            mg.add_relation(r["source"], r["target"], r.get("type", "related"), r.get("weight", 1.0))
            
        # 3. 初始化场景节点 - 基于 selection_rules
        for rule in assets_cfg.get("selection_rules", []):
            scenario_filter = rule.get("scenario_filter", {})
            # 创建唯一的场景ID
            scenario_id = compute_scenario_id(scenario_filter, prefix="rule_scenario")
            mg.add_scenario_node(scenario_id, attributes=scenario_filter)
            
            # 连接场景 -> 推荐路径 (每个推荐模型实例化为单模型 Path 超边)
            for model_id in rule.get("prefer", []):
                if mg.G.has_node(model_id):
                    path_id = mg.instantiate_path(
                        path_id=f"path::single_model::{model_id}",
                        models=[model_id],
                        strategy="single_model",
                        created_from="selection_rule",
                    )
                    mg.add_scenario_path_edge(scenario_id, path_id, performance_score=0.8)
            
        return mg

    def _combinator_trace_path(self, scenario_id: str) -> Path:
        return Path(PROJECT_ROOT) / "reports" / "traces" / f"combinator_solver_{scenario_id}.json"

    def _build_combinator_runtime_manifests(
        self,
        model_ids: List[str],
        business_domain: str = "load_forecast",
    ) -> Dict[str, ModelManifest]:
        return {
            model_id: ModelManifest(
                model_id=model_id,
                task_types=[TaskType.FORECASTING],
                business_domains=[business_domain],
                input_constraints={"features": []},
                output_schema={"yhat": "float"},
                resource_cost={},
                lifecycle_stage=ModelLifecycleStage.ACTIVE,
            )
            for model_id in model_ids
        }

    def _select_path_with_solver(
        self,
        *,
        scenario_signature: Dict[str, Any],
        region: str,
        scenario_id: str,
        available_models: List[str],
        constraints: Dict[str, float],
        model_graph: Optional[ModelGraph],
        similar_scenarios: List[Tuple[str, float]],
        actual_columns: set,
        trace_path: Optional[Path] = None,
    ):
        data_contract = DataContract(
            required_columns={"load": "float"},
            freq="H",
            min_samples=max(1, int(scenario_signature.get("_data_size", 1))),
            business_domain="load_forecast",
        )
        scenario = ScenarioDefinition(
            task_type=TaskType.FORECASTING,
            business_domain="load_forecast",
            data_contract=data_contract,
            target_schema={"yhat": "float"},
            primary_metric="RMSE",
            signature_features=list(scenario_signature.keys()),
            signature=dict(scenario_signature),
            region=region,
            scenario_id=scenario_id,
        )
        ctx = SolveContext(
            scenario=scenario,
            available_features=set(actual_columns),
            model_cols=list(available_models),
            constraints=dict(constraints),
            model_graph=model_graph,
            similar_scenarios=list(similar_scenarios or []),
            historical_scenarios=list(getattr(self, "historical_scenarios", [])),
        )
        def _env_enabled(name: str) -> bool:
            return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

        solver_kwargs: Dict[str, Any] = {}
        # 实验开关（默认全关）：与系统 B 的 MODELCOMBINE_KG_* 开关对应
        # 边权更新策略：fixed 强制不挂时序阶段；hawkes 强制挂载（无需另开 TEMPORAL 开关）
        strategy = _edge_update_strategy()
        mount_temporal = (
            model_graph is not None
            and strategy != "fixed"
            and (strategy == "hawkes" or _env_enabled("MODELCOMBINE_PIPELINE_ENABLE_TEMPORAL_RELATIONS"))
        )
        if mount_temporal:
            solver_kwargs["temporal_relation_graph"] = model_graph
        if model_graph is not None and _env_enabled("MODELCOMBINE_PIPELINE_ENABLE_CASCADE"):
            from ..core.solver.cascade import CascadeDecider

            threshold = float(
                os.environ.get("MODELCOMBINE_PIPELINE_CASCADE_UNCERTAINTY_THRESHOLD", "0.5")
            )
            solver_kwargs["cascade_decider"] = CascadeDecider(model_graph, threshold)
        if model_graph is not None and _env_enabled("MODELCOMBINE_PIPELINE_ENABLE_SUBSTITUTE"):
            solver_kwargs["allow_substitute"] = True
            solver_kwargs["substitute_graph"] = model_graph
        solver = build_solver(
            "combinator",
            manifests=self._build_combinator_runtime_manifests(available_models),
            combinator=self.model_combinator,
            **solver_kwargs,
        )
        result, trace = solver.solve(
            ctx,
            trace_path=str(trace_path) if trace_path is not None else None,
        )
        return result.get("raw", result), trace
    
    def select_models_for_region(
        self,
        region: str,
        df_train: pd.DataFrame,
        model_graph: ModelGraph,
        available_models_override: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], Dict[str, float], str, str]:
        """为特定区域选择最优模型组合（增强版：真正利用历史场景并写入图谱）"""
        region_data = df_train[df_train["region"] == region].copy()
        
        # 提取场景特征签名
        region_type = self._infer_region_type(region)
        scenario_signature = self.scenario_analyzer.extract_scenario_signature(
            region_data, region_type=region_type
        )
        
        # Phase 2: 添加数据规模信息（用于资源预测）
        scenario_signature['_data_size'] = len(region_data)
        scenario_signature['_num_features'] = len(region_data.columns) - 3  # 减去 timestamp, region, load
        
        # Phase 3: 添加事件特征
        if self.enable_phase3:
            event_features = self.event_extractor.extract_events(region_data, 'load')
            scenario_signature.update(event_features)
            print(f"  [Phase 3] 事件特征: {event_features.get('num_change_points', 0)} 个突变点, "
                  f"{event_features.get('num_anomalies', 0)} 个异常值, "
                  f"{event_features.get('num_peaks', 0)} 个峰值")
        
        # 获取可用模型（过滤掉Blender类）
        if available_models_override is None:
            all_models = model_registry.get_available_models()
            available_models = [m for m in all_models if 'blender' not in m.lower()]
        else:
            available_models = list(dict.fromkeys(available_models_override))
            if not available_models:
                raise ValueError("available_models_override must contain at least one model")
        
        # === 核心改进1：基于历史场景库的 Top-K 检索 ===
        historical_sigs = [(sid, sig) for sid, sig, _ in self.historical_scenarios]
        similar_scenarios = self.scenario_analyzer.find_similar_scenarios(
            scenario_signature, historical_sigs, top_k=3, similarity_threshold=0.5
        )
        
        if similar_scenarios:
            print(f"  [场景检索] 找到 {len(similar_scenarios)} 个相似历史场景")
            # 提取相似场景的历史最佳路径信息
            for sid, sim in similar_scenarios[:1]:  # 仅显示最相似的
                hist_rec = next((h for h in self.historical_scenarios if h[0] == sid), None)
                if hist_rec and hist_rec[2]:
                    perf = hist_rec[2]
                    if 'overall_rmse' in perf:
                        print(f"    相似度: {sim:.3f}, 历史RMSE: {perf['overall_rmse']:.4f}")
        
        # === 核心改进2：将历史数据传递给 Combinator ===
        self.model_combinator.set_historical_scenarios(self.historical_scenarios)
        
        # 生成scenario_id并添加到图谱（提前到路径选择之前）
        scenario_id = compute_scenario_id(scenario_signature, prefix=region)
        if not model_graph.G.has_node(scenario_id):
            model_graph.add_scenario_node(scenario_id, attributes={
                "region": region,
                "region_type": region_type,
                "signature": scenario_signature
            })
        
        # 将scenario_id注入到scenario_signature中（供图推理使用）
        scenario_signature['_scenario_id'] = scenario_id
        
        # 定义约束
        constraints = {
            "max_latency": 500,  # ms
            "max_resource": 10.0
        }
        
        actual_columns = set(region_data.columns)  # 提取实际数据框列名
        trace_path = self._combinator_trace_path(scenario_id)
        optimal_path, selection_trace = self._select_path_with_solver(
            scenario_signature=scenario_signature,
            region=region,
            scenario_id=scenario_id,
            available_models=available_models,
            constraints=constraints,
            model_graph=model_graph,
            similar_scenarios=similar_scenarios,
            actual_columns=actual_columns,
            trace_path=trace_path,
        )
        
        selected_models = optimal_path["models"]
        weights = optimal_path["weights"]
        path_id = optimal_path["path_id"]
        
        # 显示是否使用了历史性能
        used_hist = optimal_path.get("metrics", {}).get("used_history", False)
        hist_flag = " [基于历史]" if used_hist else ""
        print(f"  [智能调度] 选中路径: {path_id} (策略: {optimal_path['strategy']}){hist_flag}")
        print(f"  [智能调度] SelectionTrace: {trace_path}")
        
        # === 核心改进3：将路径写入图谱 ===
        # 实例化路径节点（如果尚未存在）
        model_graph.instantiate_path(path_id, selected_models, optimal_path["strategy"])
        
        # 添加场景 -> 路径的推荐边（初始权重0.5）
        model_graph.add_scenario_path_edge(scenario_id, path_id, performance_score=0.5)
        
        # 添加场景 -> 模型的直接推荐边（用于后续反馈更新，不经过 Path 超边）
        for model_id in selected_models:
            if model_graph.G.has_node(model_id):
                model_graph.add_scenario_model_edge(scenario_id, model_id, weight=0.5)
        
        # 暂存空的 performance，待评估后回填（新增path_id和weights_used字段）
        perf_init = {
            "path_id": path_id,
            "weights_used": optimal_path.get("weights_used", {"error": 0.6, "latency": 0.2, "resource": 0.2}),
            "optimize_info": optimal_path.get("optimize_info", {}),  # SLA 优化详情
            "selection_trace_path": str(trace_path),
            "selection_trace_stages": [s["stage"] for s in selection_trace.stages],
        }
        self.historical_scenarios.append((scenario_id, scenario_signature, perf_init))
        
        return selected_models, weights, scenario_id, path_id
    
    def feedback_loop(self, evaluation_results: Dict[str, Any], model_graph: ModelGraph, 
                     scenario_id_map: Dict[str, str] = None, path_id_map: Dict[str, str] = None,
                     all_model_performances: Dict[str, Dict] = None):
        """闭环反馈：根据实际误差更新图谱权重与历史场景（增强：路径级性能+profiling存储）"""
        print("执行闭环反馈学习...")
        learning_rate = 0.1
        all_model_performances = all_model_performances or {}
        
        # 遍历每个区域的评估结果
        for region, metrics in evaluation_results.get("by_region", {}).items():
            rmse = metrics.get("RMSE", 1.0)
            # 简单的分数映射：RMSE越小分数越高，限制在0-1之间
            performance_score = 1.0 / (1.0 + rmse) 
            
            print(f"  区域 {region} 反馈: RMSE={rmse:.4f}, Score={performance_score:.4f}")
            ema_enabled = _edge_update_strategy() == "moving_average"
            
            # === 核心改进1：精确匹配场景ID并更新历史库（含路径级性能） ===
            target_scenario_id = scenario_id_map.get(region) if scenario_id_map else None
            target_path_id = path_id_map.get(region) if path_id_map else None
            updated = False
            
            for i in range(len(self.historical_scenarios) - 1, -1, -1):
                sid, sig, perf = self.historical_scenarios[i]
                # 精确匹配场景ID
                if target_scenario_id and sid == target_scenario_id:
                    # 回填路径级性能数据
                    perf["overall_rmse"] = rmse
                    perf["score"] = performance_score
                    perf["path_id"] = target_path_id  # 记录使用的路径
                    perf["path_rmse"] = rmse  # 路径级RMSE（组合后的实际表现）
                    
                    # 记录具体模型的性能（模型级）
                    if "model_comparison" in evaluation_results:
                        region_models = evaluation_results["model_comparison"].get(region, {})
                        for m_name, m_metrics in region_models.items():
                            perf[m_name] = m_metrics
                    
                    # === 保存profiling数据（真实性能观测） ===
                    if region in all_model_performances:
                        region_perf = all_model_performances[region]
                        if '_profiling' in region_perf:
                            perf['_profiling'] = region_perf['_profiling']
                            print(f"  已保存性能观测数据: {region_perf['_profiling']}")
                    
                    updated = True
                    print(f"  已更新场景 {sid} 的性能记录（路径: {target_path_id}）")
                    
                    # === 核心改进2：更新图谱边权（场景->模型 和 场景->路径） ===
                    if model_graph.G.has_node(sid):
                        # 2.1 更新场景 -> 路径的边权
                        if ema_enabled and target_path_id and model_graph.G.has_edge(sid, target_path_id):
                            model_graph.update_edge_weight(sid, target_path_id, performance_score, learning_rate)
                            print(f"    已更新图谱边权: {sid} -> {target_path_id} (Score: {performance_score:.4f})")
                        
                        # 2.2 更新场景 -> 各模型的边权
                        if "model_comparison" in evaluation_results:
                            for model_id, model_metrics in region_models.items():
                                model_rmse = model_metrics.get("RMSE", 1.0)
                                model_score = 1.0 / (1.0 + model_rmse)
                                
                                if ema_enabled and model_graph.G.has_edge(sid, model_id):
                                    model_graph.update_edge_weight(sid, model_id, model_score, learning_rate)
                                    print(f"    已更新图谱边权: {sid} -> {model_id} (Score: {model_score:.4f})")
                        
                        # === Phase 2: 更新自适应权重管理器 ===
                        if self.enable_phase2 and self.model_combinator.weight_manager:
                            # 从评估结果中提取使用的权重
                            weights_used = perf.get('weights_used', {'error': 0.6, 'latency': 0.2, 'resource': 0.2})
                            
                            # 提取场景签名（移除非特征字段）
                            sig_dict = {k: v for k, v in sig.items() if not k.startswith('_')}
                            
                            # 传入 dict 格式的参数
                            self.model_combinator.weight_manager.update_history(
                                sig_dict, 
                                weights_used,  # 直接传 dict
                                {'error': rmse, 'latency': metrics.get('MAE', 0), 'resource': metrics.get('MAPE', 0)}
                            )
                            print(f"    已更新自适应权重历史")
                    
                    break
            
            if not updated and not target_scenario_id:
                # 兜底：如果没有精确ID，尝试模糊匹配（兼容旧逻辑）
                for i in range(len(self.historical_scenarios) - 1, -1, -1):
                    sid, _, perf = self.historical_scenarios[i]
                    if region in sid:
                        perf["overall_rmse"] = rmse
                        perf["score"] = performance_score
                        perf["path_rmse"] = rmse
                        if "model_comparison" in evaluation_results:
                            region_models = evaluation_results["model_comparison"].get(region, {})
                            for m_name, m_metrics in region_models.items():
                                perf[m_name] = m_metrics
                        print(f"  已更新场景 {sid} 的性能记录（模糊匹配）")
                        break
                
        # 保存更新后的历史场景库
        self._save_history()
        print(f"  历史场景库已保存至 {self.history_path}")
        
        # Phase 3: 训练场景编码器
        if self.enable_phase3 and len(self.historical_scenarios) >= 3:
            print(f"  [Phase 3] 训练场景编码器（样本数: {len(self.historical_scenarios)}）")
            scenario_list = [sig for _, sig, _ in self.historical_scenarios]
            self.scenario_encoder.fit(scenario_list)
            if self.scenario_encoder.is_fitted:
                encoder_info = self.scenario_encoder.get_encoding_info()
                print(f"  [Phase 3] 场景编码器已训练（编码维度: {encoder_info['encoding_dim']}, "
                      f"特征数: {encoder_info['n_features']}）")
            
    def _infer_region_type(self, region: str) -> str:
        """从区域名称推断区域类型"""
        region_lower = region.lower()
        if "residential" in region_lower:
            return "residential"
        elif "charging" in region_lower:
            return "charging"
        elif "service" in region_lower:
            return "service_area"
        else:
            return "residential"  # 默认类型
    
    def _predict_region_with_protocol_b(
        self,
        region: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model_graph: Optional[ModelGraph] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Protocol B 独立流程：先生成候选预测矩阵，再交统一 solver 组合。

        与 `fit_and_predict_region`（combinator 兼容路径）完全分离，互不影响：
        本方法不触碰 `historical_scenarios`、生产 `ModelGraph`、旧权重/资源模型
        或生产反馈存储——adapter 内部使用隔离的 `KGFeedbackStore`。

        Raises:
            RuntimeError: Protocol B 未使用引擎运行时精确预测（`yhat_source` 不是
                `engine`）。此时线性重建会与上报 MAE 不一致，绝不能当作输出。
        """
        import time

        pipe_cfg = self.config["pipeline"]
        # Task 8.1：默认候选池只收已获准发布的模型；被排除者原因见
        # src/utils/determinism.DEFAULT_CANDIDATE_EXCLUSIONS，并写入 trace/报告，
        # 不做静默剔除。
        from ..utils.determinism import default_candidate_models, default_exclusion_reason

        candidate_models = default_candidate_models(model_registry.get_available_models())
        excluded = {
            m: default_exclusion_reason(m)
            for m in model_registry.get_available_models()
            if default_exclusion_reason(m)
        }
        if excluded:
            print(f"   [determinism] 默认候选池已排除未获准发布的模型: {sorted(excluded)}")
        validation_days = int(pipe_cfg["data"].get("validation_days", 30))

        start = time.perf_counter()
        bundle = build_region_prediction_bundle(
            region=region,
            train=train,
            test=test,
            candidate_models=candidate_models,
            validation_days=validation_days,
            model_params=pipe_cfg.get("models", {}),
        )
        trace_path = Path(PROJECT_ROOT) / "reports" / "traces" / f"protocol_b_demo_{region}.json"
        # Protocol B 失败时直接抛出：静默切回 combinator 会让"以为在跑 B、
        # 实际是旧引擎"混进实验记录，这正是本计划要终结的问题。
        adapter_result = DemoProtocolBAdapter().select(
            bundle, region=region, horizon=1, trace_path=trace_path,
            # §11#7：把生产图谱传下去，否则关系强度在默认 run.py 路径上恒为中性
            model_graph=model_graph,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if adapter_result.get("yhat_source") != "engine":
            raise RuntimeError(
                f"Protocol B for region {region!r} did not use engine runtime predictions "
                f"(yhat_source={adapter_result.get('yhat_source')!r}); refusing to emit a "
                "linearly reconstructed yhat that may disagree with the reported MAE"
            )

        test_r = test[test["region"] == region].copy() if "region" in test.columns else test.copy()
        result = test_r[["timestamp", "region", "load"]].copy()
        result["yhat"] = np.asarray(adapter_result["yhat"], dtype=float)

        performance = {
            "_profiling": {"elapsed_seconds": round(elapsed_ms / 1000.0, 4)},
            "_determinism": {"excluded_models": excluded},
            "_protocol_b": {
                "strategy": adapter_result.get("strategy"),
                "models": adapter_result.get("models"),
                "weights": adapter_result.get("weights"),
                "mae": adapter_result.get("mae"),
                "yhat_source": adapter_result.get("yhat_source"),
                "linear_reconstruction_match": adapter_result.get("linear_reconstruction_match"),
            },
        }
        adapter_result = dict(adapter_result)
        adapter_result["elapsed_ms"] = elapsed_ms
        return result, performance, adapter_result

    def fit_and_predict_region(self, region: str, selected_models: List[str],
                             weights: Dict[str, float], train: pd.DataFrame,
                             test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """为特定区域训练模型并预测（增强版：真实性能观测）"""
        import time
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False
        
        # 记录开始时的资源状态
        start_time = time.perf_counter()
        if has_psutil:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        train_r = train[train["region"] == region].copy()
        test_r = test[test["region"] == region].copy()
        
        if len(train_r) == 0 or len(test_r) == 0:
            print(f"警告: 区域 {region} 没有足够的训练或测试数据")
            return pd.DataFrame(), {}
        
        X_train, y_train = build_matrix(train_r)
        X_test, y_test = build_matrix(test_r)
        
        # 训练各个模型
        fitted_models = {}
        model_performances = {}
        pipe_cfg = self.config["pipeline"]
        
        for model_id in selected_models:
            try:
                # 获取模型参数
                params = pipe_cfg.get("models", {}).get(model_id, {})
                
                # 创建并训练模型
                model = model_registry.create(model_id, **params)
                model.fit(X_train, y_train)
                fitted_models[model_id] = model
                
                # 评估模型性能
                train_pred = model.predict(X_train)
                train_metrics = evaluate_all(y_train.values, train_pred)
                model_performances[model_id] = train_metrics
                
                print(f"模型 {model_id} 训练完成，训练集RMSE: {train_metrics['RMSE']:.4f}")
                
            except Exception as e:
                print(f"模型 {model_id} 训练失败: {str(e)}")
                continue
        
        if not fitted_models:
            print(f"区域 {region} 没有成功训练的模型")
            return pd.DataFrame(), {}
        
        # 创建集成模型
        ensemble_method = pipe_cfg.get("selection", {}).get("combiner", "weighted")
        ensemble = self.model_combinator.create_ensemble(
            list(fitted_models.keys()), weights, ensemble_method
        )
        
        # 训练集成模型
        if ensemble_method == "stacking":
            ensemble.fit(fitted_models, X_train, y_train)
        else:
            ensemble.fit(fitted_models, X_train, y_train)
        
        # 预测
        yhat = ensemble.predict(X_test)
        
        # 构建结果DataFrame
        result = test_r[["timestamp", "region", "load"]].copy()
        result["yhat"] = yhat
        
        # 添加个体模型预测（用于分析）
        for model_id, model in fitted_models.items():
            result[f"yhat_{model_id}"] = model.predict(X_test)
        
        # === 真实性能观测 ===
        elapsed = time.perf_counter() - start_time
        profiling_data = {
            'elapsed_seconds': round(elapsed, 4)
        }
        
        if has_psutil:
            mem_after = process.memory_info().rss / 1024 / 1024
            profiling_data['memory_mb'] = round(mem_after - mem_before, 2)
            profiling_data['cpu_percent'] = round(psutil.cpu_percent(interval=0.1), 2)
        
        # 将性能观测添加到返回的性能字典中
        model_performances['_profiling'] = profiling_data
        print(f"  [性能观测] 耗时: {profiling_data['elapsed_seconds']}s" + 
              (f", 内存: {profiling_data.get('memory_mb', 0):.1f}MB" if has_psutil else ""))
        
        # Phase 2: 将真实性能观测反馈给资源预测器
        if self.enable_phase2 and self.model_combinator.resource_predictor:
            data_size = len(train_r)
            num_features = X_train.shape[1]
            latency_ms = profiling_data['elapsed_seconds'] * 1000
            memory_mb = profiling_data.get('memory_mb', 0)
            cpu_percent = profiling_data.get('cpu_percent', 0)
            
            for model_id in selected_models:
                self.model_combinator.resource_predictor.add_observation(
                    model_id, data_size, num_features, 
                    latency_ms, memory_mb, cpu_percent
                )
        
        return result, model_performances


    def evaluate_results(self, df_pred: pd.DataFrame, 
                        model_performances: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """评估预测结果"""
        metrics_by_region = {}
        model_comparison = {}
        
        for region, g in df_pred.groupby("region"):
            y_true = g["load"].values
            y_pred = g["yhat"].values
            
            # 整体集成模型评估
            region_metrics = evaluate_all(y_true, y_pred)
            metrics_by_region[region] = region_metrics
            
            # 个体模型评估
            individual_metrics = {}
            for col in g.columns:
                if col.startswith("yhat_") and col != "yhat":
                    model_name = col.replace("yhat_", "")
                    model_pred = g[col].values
                    individual_metrics[model_name] = evaluate_all(y_true, model_pred)
                    
            model_comparison[region] = individual_metrics
        
        # 整体评估
        overall_metrics = evaluate_all(df_pred["load"].values, df_pred["yhat"].values)
        
        return {
            "overall": overall_metrics,
            "by_region": metrics_by_region,
            "model_comparison": model_comparison,
            "training_performance": model_performances,
            "summary": self._generate_summary(overall_metrics, metrics_by_region)
        }
    
    def _generate_summary(self, overall_metrics: Dict[str, float], 
                         region_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """生成预测结果摘要"""
        summary = {
            "overall_performance": {
                "RMSE": overall_metrics["RMSE"],
                "MAE": overall_metrics["MAE"],
                "MAPE": overall_metrics["MAPE"]
            },
            "best_region": None,
            "worst_region": None,
            "region_count": len(region_metrics),
            "avg_rmse_by_region": 0.0
        }
        
        if region_metrics:
            # 找出最佳和最差区域
            region_rmse = {region: metrics["RMSE"] for region, metrics in region_metrics.items()}
            summary["best_region"] = min(region_rmse, key=region_rmse.get)
            summary["worst_region"] = max(region_rmse, key=region_rmse.get)
            summary["avg_rmse_by_region"] = np.mean(list(region_rmse.values()))
            
        return summary
    
    def run_prediction_pipeline(self) -> Dict[str, Any]:
        """运行完整的电力预测流水线"""
        print("开始电力需求预测流水线...")
        
        # 1. 确保数据存在
        print("1. 检查和生成数据...")
        self.ensure_power_data()
        
        # 2. 构建特征
        print("2. 构建特征...")
        df = self.build_power_features()
        print(f"特征构建完成，数据形状: {df.shape}")
        
        # 3. 分割训练测试集
        print("3. 分割训练测试集...")
        pipe_cfg = self.config["pipeline"]
        train, test = self.split_train_test(df, test_days=pipe_cfg["data"]["test_days"])
        print(f"训练集: {len(train)} 条记录，测试集: {len(test)} 条记录")
        
        # 4. 构建模型图谱
        print("4. 构建模型关系图谱...")
        model_graph = self.build_model_graph()
        
        # 5. 为每个区域进行预测
        print("5. 开始区域预测...")
        all_predictions = []
        all_model_performances = {}
        
        regions = pipe_cfg["data"]["regions"]
        # 存储场景ID和路径ID映射，用于反馈时精确匹配
        scenario_id_map = {}
        path_id_map = {}
        
        # 三态后端（Task 3/4）：combinator 为迁移期默认；shadow 两条都跑但输出仍取
        # combinator；protocol_b 输出取 Protocol B。未知值在 resolve 处已报错。
        backend_mode = self.backend_mode
        backend_report: Dict[str, Any] = {"mode": backend_mode, "regions": {}}
        run_combinator = backend_mode in (BACKEND_COMBINATOR, BACKEND_PROTOCOL_B_SHADOW)
        run_protocol_b = backend_mode in (BACKEND_PROTOCOL_B, BACKEND_PROTOCOL_B_SHADOW)
        _explicit = bool(os.environ.get("MODELCOMBINE_PIPELINE_BACKEND", "").strip())
        print(
            f"   决策后端: {backend_mode}"
            f"（{'显式指定' if _explicit else '默认'}；默认值为 {BACKEND_PROTOCOL_B}）"
        )
        if backend_mode == BACKEND_COMBINATOR:
            print("   [注意] 正在使用迁移期回退路径 combinator（旧 System A 决策引擎）")

        for i, region in enumerate(regions, 1):
            print(f"\n处理区域 {i}/{len(regions)}: {region}")

            combinator_pred = None
            combinator_performance = None
            combinator_result = None
            combinator_elapsed_ms = None
            if run_combinator:
                import time as _time

                _t0 = _time.perf_counter()
                # 选择模型（现在返回 scenario_id 和 path_id）
                selected_models, weights, scenario_id, path_id = self.select_models_for_region(region, train, model_graph)
                scenario_id_map[region] = scenario_id
                path_id_map[region] = path_id

                print(f"选择的模型: {selected_models}")
                print(f"模型权重: {weights}")

                # 训练和预测
                combinator_pred, combinator_performance = self.fit_and_predict_region(
                    region, selected_models, weights, train, test
                )
                combinator_elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
                combinator_result = {
                    "models": list(selected_models),
                    "weights": dict(weights),
                    "strategy": "combinator",
                    "mae_recomputed": None,
                }

            protocol_b_result = None
            protocol_b_pred = None
            protocol_b_performance = None
            if run_protocol_b:
                protocol_b_pred, protocol_b_performance, protocol_b_result = (
                    self._predict_region_with_protocol_b(
                        region, train, test, model_graph=model_graph
                    )
                )

            if backend_mode == BACKEND_PROTOCOL_B:
                region_pred, region_performance = protocol_b_pred, protocol_b_performance
                # 用 Protocol B 上下文的真实 scenario_id 参与反馈：
                # 此前写死 protocol_b::{region}，与消费侧按签名派生的场景 id
                # 不一致，反馈边永远落在一个没人读的场景上。
                scenario_id_map[region] = protocol_b_result.get("scenario_id")
                path_id_map[region] = protocol_b_result.get("path_id")
                region_report: Dict[str, Any] = {
                    "final_output_from": "protocol_b",
                    "trace_path": str(
                        Path(PROJECT_ROOT) / "reports" / "traces" / f"protocol_b_demo_{region}.json"
                    ),
                    "yhat_source": protocol_b_result.get("yhat_source"),
                    "strategy": protocol_b_result.get("strategy"),
                    "models": protocol_b_result.get("models"),
                    "mae": protocol_b_result.get("mae"),
                    "linear_reconstruction_match": protocol_b_result.get(
                        "linear_reconstruction_match"
                    ),
                }
            else:
                region_pred, region_performance = combinator_pred, combinator_performance
                region_report = {"final_output_from": "combinator"}
                if backend_mode == BACKEND_PROTOCOL_B_SHADOW:
                    # 影子模式只产审计：最终输出仍是 combinator，逐值不受影响。
                    region_report["yhat_source"] = protocol_b_result.get("yhat_source")
                    region_report["trace_path"] = str(
                        Path(PROJECT_ROOT) / "reports" / "traces" / f"protocol_b_demo_{region}.json"
                    )
                    region_report["comparison"] = build_shadow_comparison(
                        combinator_result=combinator_result,
                        protocol_b_result={
                            **protocol_b_result,
                            "mae_recomputed": protocol_b_result.get("mae"),
                        },
                        combinator_elapsed_ms=combinator_elapsed_ms,
                        protocol_b_elapsed_ms=protocol_b_result.get("elapsed_ms"),
                    )
            backend_report["regions"][region] = region_report

            if region_pred is not None and not region_pred.empty:
                all_predictions.append(region_pred)
                all_model_performances[region] = region_performance
        
        # 6. 合并结果
        print("\n6. 合并预测结果...")
        if all_predictions:
            pred_df = pd.concat(all_predictions, ignore_index=True)
        else:
            print("警告: 没有成功的预测结果")
            return {"error": "No successful predictions"}
        
        # 7. 评估结果
        print("7. 评估预测性能...")
        evaluation_results = self.evaluate_results(pred_df, all_model_performances)
        
        # 8. 闭环反馈
        print("8. 闭环反馈学习...")
        self.feedback_loop(evaluation_results, model_graph, scenario_id_map, path_id_map, all_model_performances)
        
        # 9. 保存结果
        print("9. 保存结果...")
        out_dir = os.path.join(PROJECT_ROOT, "reports")
        ensure_dir(out_dir)
        
        # 保存预测结果
        pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        
        # 保存评估报告
        save_json(evaluation_results, os.path.join(out_dir, "report.json"))
        
        # 保存详细的模型信息
        model_info = {
            "pipeline_config": self.config,
            "data_shape": df.shape,
            "train_test_split": {
                "train_size": len(train),
                "test_size": len(test),
                "test_days": pipe_cfg["data"]["test_days"]
            },
            "regions_processed": regions,
            "available_models": model_registry.get_available_models(),
            # Task 8：记录本次实际使用的决策后端。fell_back_to_combinator 只在用户
            # 显式设置 combinator 时为真——Protocol B 失败会直接抛错，不存在静默回退。
            "backend": {
                "mode": backend_mode,
                "default_mode": BACKEND_PROTOCOL_B,
                "is_shadow": backend_mode == BACKEND_PROTOCOL_B_SHADOW,
                "fell_back_to_combinator": backend_mode == BACKEND_COMBINATOR,
                "explicitly_set": bool(
                    os.environ.get("MODELCOMBINE_PIPELINE_BACKEND", "").strip()
                ),
                "regions": backend_report.get("regions", {}),
            },
        }
        save_json(model_info, os.path.join(out_dir, "model_info.json"))
        
        print(f"\n预测完成！结果保存在 {out_dir}")
        print(f"整体RMSE: {evaluation_results['overall']['RMSE']:.4f}")
        print(f"整体MAE: {evaluation_results['overall']['MAE']:.4f}")
        print(f"整体MAPE: {evaluation_results['overall']['MAPE']:.4f}%")
        
        # === 10. 持久化图谱状态（跨进程保留边权） ===
        print("10. 保存图谱状态...")
        graph_persist_path = os.environ.get("MODELCOMBINE_GRAPH_STATE_PATH") or os.path.join(out_dir, "graph_state.pkl")
        model_graph.save_graph(graph_persist_path)
        
        # === 11. Phase 2: 保存轻量级学习模型 ===
        if self.enable_phase2:
            print("11. 保存 Phase 2 学习模型...")
            self._save_phase2_models()
        
        # === 12. Phase 3: 保存高级特征模型 ===
        if self.enable_phase3:
            print("12. 保存 Phase 3 学习模型...")
            self._save_phase3_models()

        # 决策后端审计信息随评估结果返回。注意 report.json 在第 9 步已保存，
        # 此处不会污染其内容。
        evaluation_results["backend"] = backend_report
        # 仅非默认模式落盘审计文件：combinator 默认路径的产物集合必须保持不变。
        if backend_mode != BACKEND_COMBINATOR:
            save_json(backend_report, os.path.join(out_dir, "backend_report.json"))

        return evaluation_results


# 保持向后兼容的函数
def load_configs() -> Dict[str, Any]:
    assets = load_asset_yaml(os.path.join(PROJECT_ROOT, "configs", "model_assets.yaml"))
    pipe = load_yaml(os.path.join(PROJECT_ROOT, "configs", "pipeline.yaml"))
    return {"assets": assets, "pipeline": pipe}


def ensure_input_data(pipe_cfg: Dict[str, Any]) -> None:
    """向后兼容的数据检查函数。"""
    pipeline = PowerPredictionPipeline()
    pipeline.ensure_power_data()


def build_features(pipe_cfg: Dict[str, Any]) -> pd.DataFrame:
    """向后兼容的特征构建函数"""
    pipeline = PowerPredictionPipeline()
    return pipeline.build_power_features()


def run_pipeline() -> None:
    """运行电力预测流水线"""
    pipeline = PowerPredictionPipeline()
    results = pipeline.run_prediction_pipeline()
    return results


# --- SQLite 模型库在线入口（不接回 System A）--------------------------------

def library_predict(*, database: str, scenario: str, features: str, output: str) -> Dict[str, Any]:
    """从模型库匹配历史关系、加载产物并对用户未来特征产生预测。"""
    from src.pipeline.library_prediction import predict

    return predict(
        database=database,
        scenario_path=scenario,
        features_path=features,
        output_path=output,
    )


def library_feedback(*, database: str, prediction_run_id: int, actual: str) -> Dict[str, Any]:
    """用后来返回的真实值更新对应关系的实际表现统计。"""
    from src.pipeline.library_feedback import apply_feedback

    return apply_feedback(
        database=database,
        prediction_run_id=int(prediction_run_id),
        actual_path=actual,
    )


if __name__ == "__main__":
    run_pipeline()
