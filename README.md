# 基于知识图谱与场景相似度的电力多模型动态调度系统

## 项目概述

Modelcombine 是一个基于知识图谱与场景相似度的电力多模型动态调度方法及系统。系统将电力业务场景、数据资产、特征、模型能力、模型关系、组合路径与历史性能组织为知识图谱，并结合场景相似度检索历史经验，在误差、时延、资源等约束下动态选择单模型或模型组合，实现面向不同电力业务场景的自适应模型调度。

- **方法层**：基于知识图谱与场景相似度的多模型动态调度方法
- **系统层**：面向电力业务场景的多模型动态调度系统
- **验证层**：当前以电力负荷预测为主要实验场景（电费回收风险预测、设备故障预测、异常用电识别等为规划中的扩展场景）

### 核心特性

🔋 **多场景支持**: 住宅用电、充电桩用电、服务区用电等多种场景
🤖 **智能模型组合**: 自动选择和组合最适合的预测模型
📊 **多模态数据融合**: 整合用电数据、天气数据、时间特征等
🎯 **场景相似度匹配**: 基于历史经验智能选择模型策略
📈 **实时预测评估**: 提供RMSE、MAE、MAPE等多种评估指标

## 目录结构
```
.
├─ data/                # 示例数据（首次为空，提供生成脚本）
├─ configs/             # 配置（示例：model_assets.yaml, pipeline.yaml）
├─ reports/             # 预测与评估报告输出
├─ src/
│  ├─ data/             # 数据加载、验证、示例生成
│  ├─ features/         # 特征工程与跨模态融合（天气+时间+历史统计）
│  ├─ models/           # 模型封装、训练与预测接口
│  ├─ graph/            # 模型图谱构建与查询
│  ├─ selector/         # 场景相似度、路径挖掘与组合策略
│  ├─ pipeline/         # 端到端流程控制（train/predict/eval）
│  └─ utils/            # 公共工具（metrics、io）
├─ requirements.txt
└─ README.md
```

## 快速开始
1. 安装依赖（建议使用虚拟环境）
2. 运行示例生成与主流程：
   - 生成合成数据
   - 一键训练、预测、评估

详见下文“如何运行”。

## 背景与目标
围绕电力客户“用电需求智能预测”，以“模型图谱+场景相似度+组合策略”为核心，面对多区域/多模态数据，自动选择最优模型组合，提高预测准确性与适应性。

## 模块设计概述
- 数据层：`src/data/` 提供原始多区域多模态（用电+天气+节假日）数据读写与合成数据。
- 特征层：`src/features/` 提取时间、节假日、天气融合特征，以及简单的过去窗口统计特征。
- 模型资产：`configs/model_assets.yaml` 描述可用模型、指标与IO规范；`src/models/registry.py` 进行注册。
- 图谱层：`src/graph/` 使用 NetworkX 构建模型关系：依赖、互补、竞争，支持查询关联路径。
- 相似度与选择：`src/selector/` 基于场景（区域类型/分布/季节性）相似度检索历史最佳组合，并沿图谱路径挖掘候选组合。
- 组合与集成：加权平均、Stacking 两种简单策略，便于扩展。
- 流水线：`src/pipeline/main.py` 统一入口：load -> feature -> select -> train -> predict -> eval -> report。

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd Modelcombine

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行预测系统

```bash
# 直接运行主程序
python run.py

# 或者使用模块方式运行
python -m src.pipeline.main
```

### 3. 查看结果

运行完成后，结果将保存在 `reports/` 目录下：
- `predictions.csv`: 详细的预测结果
- `report.json`: 完整的评估报告
- `model_info.json`: 模型配置和运行信息
- `traces/`: 模型动态调度的 `SelectionTrace`，记录能力匹配、相似场景、组合后端和不确定性/级联阶段

## 指令清单（常用）

### 环境与依赖
- 安装依赖：`python -m pip install -r requirements.txt`

### 数据准备与特征
- 切分数据：`python scripts/split_datasets.py`
- 生成特征：`python scripts/generate_features.py`
- 下载/补充数据：`python scripts/download_additional_data.py`

### 训练与评估
- 训练基线模型（含 prophet/arima 等外部模型预测）：`python scripts/train_baselines.py --out reports/baselines`
- 组合策略评估：`python scripts/modelcombine_eval.py --baselines reports/baselines`
- 统一对比报告（Baseline/KG Core/KG Component）：`python scripts/compare_results.py --baseline-root reports/modelcombine --kg-root reports/combos_kg --leaderboard-root reports/modelcombine --out reports/analysis/comparison_report.md`
- 端到端运行：`python run.py`

### 结果可视化
- 生成热力图（基线/组合）：`python scripts/generate_heatmap.py --result-dir result`
- 仅组合策略（KG 组件）：`python scripts/generate_heatmap.py --metrics result/0205/metrics(4).json --kind combos --group kg_component`
- 组合相对最优单模型提升：`python scripts/generate_heatmap.py --metrics result/0205/metrics(4).json --kind combos --relative best_base`

## 系统架构

### 数据流程
```
原始数据 → 特征工程 → 场景分析 → 模型选择 → 模型训练 → 预测评估 → 结果输出
```

### 核心模块

1. **数据采集模块** (`src/data/`)
   - 电力负荷数据采集
   - 天气数据采集
   - 节假日数据处理

2. **特征工程模块** (`src/features/`)
   - 时间特征提取
   - 滞后特征构建
   - 多模态特征融合

3. **模型管理模块** (`src/models/`)
   - 时序预测模型（Prophet、ARIMA）
   - 回归模型（XGBoost、LightGBM、CatBoost）
   - 电力专用模型（差异分析、多模态融合）

4. **智能选择模块** (`src/selector/`)
   - 场景相似度计算
   - 模型组合策略
   - 历史经验学习

5. **预测流水线** (`src/pipeline/`)
   - 端到端预测流程
   - 自动化模型训练
   - 结果评估和报告

## 模型组合策略命名约定（统一口径）

为避免“本项目方法”概念混淆，结果统一采用两层命名：

- `strategy`：稳定机器名（兼容历史脚本）
- `strategy_display_name` / `method_family` / `method_route` / `core_route`：语义名

### 1) KG Core（固定主路线：Knowledge Graph）
- **kg_protocol_a**（兼容旧键 `protocol_A`）：
  基于图谱关系与预测误差推理，使用预测侧信息完成组合。
- **kg_protocol_b**（兼容旧键 `protocol_B`）：
  在 A 基础上引入原始特征侧信号进行增强。

说明：`core_route=yes` 仅对应 KG Protocol A/B。

### 2) KG Component（非主路线，对照/消融）
- **gating_network / soft_gating / gating_network_v2**
- **scenario_bucket / adaptive_bucket / scenario_similarity**

说明：这些方法用于验证不同动态组合机制，统一标记为 `method_route=ablation`。

### 3) Baseline（对比基线）
- **baseline_classic**：`simple_avg`、`static_weight`、`stacking`、`dynamic_*`、`constrained_opt`
- **baseline_sota**：`rl_qms`、`mole_router`

## 支持的预测模型

### 时序预测模型
- **Prophet**: 适合处理季节性和节假日效应，特别适用于住宅和服务区用电预测
- **ARIMA**: 经典统计时序模型，适合短期预测和基线对比

### 机器学习模型
- **XGBoost**: 梯度提升树，适合特征丰富的复杂场景，如充电桩用电预测
- **LightGBM**: 高效梯度提升模型，适合大规模数据和快速训练
- **CatBoost**: 对类别特征友好，适合包含区域类型等分类特征的场景

### 电力专用模型
- **电力负荷差异分析模型**: 专门处理不同区域间的用电差异和峰谷特征
- **多模态数据融合模型**: 整合用电、天气、时间等多种数据源的融合预测

## 配置说明

### 模型配置 (`configs/model_assets.yaml`)
定义可用模型、模型关系和选择规则：
```yaml
models:
  - id: "prophet"
    type: "ts_forecast"
    family: "bayesian_additive"
    best_for: ["强季节性", "节假日效应", "住宅用电"]
```

### 流水线配置 (`configs/pipeline.yaml`)
配置数据处理、特征工程和模型参数：
```yaml
data:
  regions: ["residential_a", "charging_a", "service_area_a"]
  freq: "H"
  test_days: 7

features:
  lags: [1, 2, 24, 168]  # 滞后特征
  rolling:
    - { window: 24, stat: "mean" }
```

### 组合策略环境变量
以下环境变量用于快速调参（不改代码）：

- `MODELCOMBINE_KG_COMPONENT_POOL_MODE`
  - 可选：`safe`（默认）/ `scene_fit`
  - 作用：控制 KG component 策略使用的模型池（与 baseline 对齐或场景受限池）
- `MODELCOMBINE_DIRECT_WEIGHT_TEMPERATURE`
  - 默认：`0.5`
  - 作用：`DirectWeightGatingNetwork` 的 softmax 温度
- `MODELCOMBINE_SCENARIO_SIM_WEIGHT_MODE`
  - 可选：`error_softmax`（默认）/ `rank` / `softmax`
  - 作用：`ScenarioSimilarityEnhancer` 的样本权重模式
- `MODELCOMBINE_SCENARIO_SIM_TEMPERATURE`
  - 默认：`2.0`
  - 作用：`ScenarioSimilarityEnhancer` 的温度参数
- `MODELCOMBINE_DRIFT_DECAY_BASE`
  - 默认：`0.001`
  - 作用：漂移感知时间衰减的基础项
- `MODELCOMBINE_DRIFT_DECAY_SLOPE`
  - 默认：`0.005`
  - 作用：漂移感知时间衰减的 PSI 斜率项
- `MODELCOMBINE_DRIFT_DECAY_MAX`
  - 默认：`0.05`
  - 作用：漂移感知时间衰减的上限

示例：
```bash
MODELCOMBINE_KG_COMPONENT_POOL_MODE=safe \
MODELCOMBINE_SCENARIO_SIM_WEIGHT_MODE=error_softmax \
MODELCOMBINE_DRIFT_DECAY_SLOPE=0.005 \
python scripts/modelcombine_eval.py --baselines reports/baselines
```

## 技术特点

### 1. 智能模型选择
- 基于场景特征自动选择最适合的模型组合
- 支持规则驱动、性能驱动和特征驱动的选择策略
- 历史场景相似度匹配和经验学习

### 2. 多模态数据融合
## 热力图可视化（基线 vs 组合）

支持自动从 `metrics*.json` 生成热力图（基础模型/组合策略），用于快速对比不同 horizon 的效果。

**示例：**
- 生成最新结果的全部热力图：
   - `python scripts/generate_heatmap.py --result-dir result`
- 仅生成组合策略，且只看 KG component：
   - `python scripts/generate_heatmap.py --metrics result/0205/metrics(4).json --kind combos --group kg_component`
- 生成组合相对最优单模型的提升百分比：
   - `python scripts/generate_heatmap.py --metrics result/0205/metrics(4).json --kind combos --relative best_base`

输出位置：`reports/heatmaps/`（同时生成 CSV 与 PNG）
- 用电负荷数据：历史用电量、峰谷特征、负荷因子
- 天气数据：温度、湿度、风速、体感温度、舒适度指数
- 时间特征：小时、星期、月份、季节、节假日标记
- 区域特征：区域类型、地理位置、用电模式

### 3. 场景相似度计算
- 多维度特征签名提取
- 加权欧几里得距离、余弦相似度等多种相似度度量
- 基于相似历史场景的模型性能预测

### 4. 模型组合策略
- 加权平均：基于历史性能和场景特征的动态权重分配
- Stacking集成：使用元学习器组合多个基础模型
- 支持自定义组合策略扩展

## 当前实现状态与技术权衡

本系统在核心功能上已完成P0-P1优先级的实现，同时采用了一些**工程简化策略**以快速验证知识图谱驱动的架构。以下是关键技术点的现状说明：

### ✅ 已实现功能
- **知识图谱驱动**: 场景→特征→模型→路径的多跳推理链
- **动态路径生成**: 冷启动场景下自动注册单模型/组合路径
- **闭环反馈学习**: 基于真实性能更新图谱边权重
- **多目标优化**: 误差/延迟/资源的固定权重打分 (60/20/20)
- **资源观测**: 实时采集运行耗时/内存/CPU并持久化
- **历史场景匹配**: Region Type (50%) + 特征签名欧氏距离 (50%)

### ⚠️ 技术简化说明

#### 1. SLA/多目标优化
**当前实现**: 固定权重打分 `Score = 0.6×Error + 0.2×Latency + 0.2×Resource`  
**技术限制**:
- Pareto 前沿探索与拉格朗日乘数法均已实现（见 `src/selector/sla_optimizer.py`），默认通过 `configs/pipeline.yaml: phase3.sla_optimizer.enable=false` 关闭；开启后即启用非支配解筛选与 SLSQP 约束优化
- 权重仍为固定默认（0.6/0.2/0.2），场景自适应权重为后续工作
- 权重固定，无法根据场景动态调整优先级

**采用原因**: 快速原型验证，易于调试和解释  
**改进路径**: Phase 2 引入 NSGA-II/MOEA-D，Phase 3 支持用户交互式权重调整  
**详见**: `docs/0701/plan_phase0_phase1.md` 第1节

#### 2. 资源估算
**当前实现**: 历史观测优先 + 元数据映射回退  
**技术限制**:
- 元数据使用三档标注 (`low/medium/high`)，粒度粗
- 无动态回归模型（未考虑数据规模/特征维度影响）
- 新模型冷启动时估算误差较大

**采用原因**: 避免过早引入复杂建模，优先利用真实观测数据  
**改进路径**: Phase 1 引入资源回归模型，Phase 2 部署在线性能监控  
**详见**: `docs/0701/plan_phase0_phase1.md` 第2节

#### 3. 场景编码
**当前实现**: 手工统计特征提取  
**特征集**: `mean_load`, `peak_valley_ratio`, `seasonal_amplitude`, `load_temp_corr` 等  
**技术限制**:
- 无深度嵌入（未使用 Transformer/LSTM 自动特征学习）
- 缺失事件特征（极端天气预警、节假日类型、突发异常）
- 表达能力受限，无法捕捉复杂非线性模式

**采用原因**: 
- 当前使用合成数据，深度学习需要大规模真实样本
- 手工特征更可解释，便于业务理解和调试
- 在线推理需要低延迟，统计特征计算速度快

**改进路径**: Phase 1 集成外部事件源，Phase 2 使用 Autoencoder 学习嵌入  
**详见**: `docs/0701/plan_phase0_phase1.md` 第3节

### 📊 工程权衡决策

| 维度 | 当前选择 | 未来方向 | 触发条件 |
|------|---------|---------|---------|
| **多目标优化** | 固定权重 | Pareto 前沿 | 用户需要多方案对比 |
| **资源估算** | 元数据映射 | 回归模型 | 累积 >5000 真实样本 |
| **场景编码** | 统计特征 | 深度嵌入 | 离线实验提升 >10% |
| **推理引擎** | 图遍历 | 强化学习 | 需要动态策略优化 |

### 🚀 技术演进路线图

```
Phase 1 (当前): 统计特征 + 固定权重 + 元数据回退
       ↓
Phase 2 (3-6月): 轻量级学习 + 自适应权重 + 资源回归
       ↓
Phase 3 (6-12月): 深度嵌入 + Pareto 优化 + 在线学习
```

**完整技术限制文档**: 参见 `docs/0701/plan_phase0_phase1.md`  
**实现细节注释**: 代码中关键位置已添加 `[技术限制]` 标注块

## 扩展方向

- 🧠 引入深度学习时序模型（LSTM、Transformer、TFT）
- 📏 更丰富的相似度度量（DTW、Fréchet距离、KS检验）
- 🕸️ 更复杂的模型图谱推理（路径评分、因果约束）
- 🔄 在线学习与概念漂移检测
- ☁️ 分布式训练和云端部署支持
- 📱 实时预测API和可视化界面

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
